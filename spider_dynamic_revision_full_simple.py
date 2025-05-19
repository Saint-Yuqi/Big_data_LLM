#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Incremental retrieval-based Text-to-SQL pipeline using DeepSeek (nuwaapi.com),
with speed-oriented optimizations:
  1) Simplify each question exactly once, cached in repo_cache.json
  2) K1=2, K2=2 retrieval => total 4 examples
  3) MAX_TURNS=2 dynamic revision
  4) main LLM calls use max_tokens=128, temperature=0.3
  5) prompt cache for LLM calls => repeated prompts skip HTTP calls
  6) text_factory='replace' to avoid UTF-8 decode errors

Usage:
  python spider_dynamic_revision_full.py --build_repo
  python spider_dynamic_revision_full.py --run_dev
  python spider_dynamic_revision_full.py --run_test
"""

import os, sys, json, sqlite3, time, argparse, subprocess, hashlib, re, random
from collections import deque
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb, dotenv
import requests
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv(override=True)

#####################################
# 0) Hyperparams
#####################################
MAIN_TEMPERATURE     = 0.3   # lower => more concise/faster
SIMPLIFY_TEMPERATURE = 1.0   # can keep a bit higher for variety
K1, K2               = 2, 2  # fewer demos => faster
MAX_TURNS            = 2     # dynamic revision attempts

API_KEY  = "sk-BgjcMpvoICN0CBXK4d30NCyeToTbCuUbvJrfmLJQ5YHnFdom"
API_URL  = "https://api.nuwaapi.com/v1/chat/completions"
MODEL_NAME = "gpt-4o-mini-2024-07-18"

print("LLM's API_KEY =", API_KEY)

#####################################
# 1) Paths & constants
#####################################
ROOT        = Path(__file__).resolve().parent
SPIDER_DIR  = Path(os.getenv("SPIDER_DIR", "~/Big_data/spider_data")).expanduser()
#DB_DIR      = SPIDER_DIR / "database"
TABLE_JSON  = SPIDER_DIR / "tables.json"
EVAL_PY     = "spider/evaluation.py"
EMBED_FILE  = SPIDER_DIR / "train_embeddings.npy"
CACHE_DB    = ROOT / "prompt_cache.sqlite"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

WB_ENTITY   = "saint-yangyuqi-university-of-z-rich"
WB_PROJECT  = "text2sql-project"

#####################################
# 2) DB with text_factory
#####################################
_db = {}
def conn(db_id):
    if db_id not in _db:
        path = DB_DIR / db_id / f"{db_id}.sqlite"
        c = sqlite3.connect(path)
        c.text_factory = lambda b: b.decode(errors="replace")  # skip invalid UTF-8
        _db[db_id] = c
    return _db[db_id]

def schema(db_id):
    c= conn(db_id).cursor()
    t= [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out=[]
    for tb in t:
        if tb.startswith("sqlite"): continue
        cols= [col[1] for col in c.execute(f"PRAGMA table_info('{tb}')")]
        short= ", ".join(cols[:10]) + (", …" if len(cols)>10 else "")
        out.append(f"{tb}({short})")
    return " ; ".join(out)

def execute(db_id, sql):
    c= conn(db_id).cursor()
    c.execute(sql)
    return c.fetchall()

def equal(a, b):
    return sorted(map(str,a))== sorted(map(str,b))

#####################################
# 3) LLM calls: DeepSeek
#####################################
def call_deepseek(messages, temperature=0.7, max_tokens=256):
    """
    Post to nuwaapi.com/v1/chat/completions with model=deepseek-r1
    messages: list of {"role":"system"|"user", "content":...}
    """
    headers= {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json"
    }
    data= {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp= requests.post(API_URL, headers=headers, json=data, timeout=120)
    if resp.status_code==200:
        j= resp.json()
        return j["choices"][0]["message"]["content"].strip()
    else:
        es= resp.text.lower()
        raise RuntimeError(f"DeepSeek error {resp.status_code}: {resp.text}")

#####################################
# 4) Prompt cache
#####################################
cache_conn= sqlite3.connect(CACHE_DB)
cache_conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
cache_conn.commit()

def cache_get(h):
    row= cache_conn.execute("SELECT v FROM cache WHERE k=?",(h,)).fetchone()
    return None if row is None else row[0]

def cache_put(h,v):
    cache_conn.execute("INSERT OR REPLACE INTO cache VALUES(?,?)",(h,v))
    cache_conn.commit()

#####################################
# 5) SBERT & data
#####################################
model_sbert, train_pairs, train_vecs= None, None, None

def skeleton(text:str):
    return " ".join([w.lower() for w in text.split() if re.match(r"^[A-Za-z_]+$", w)])

def _load_json(name:str):
    import json
    with open(SPIDER_DIR/f"{name}.json","r") as f:
        return json.load(f)

#####################################
# 6) build_repo
#####################################
def build_repo(limit=None):
    """
    We do question simplification once per unique question, store in simplified_map,
    and store skeleton->SQL pairs in 'items'.
    Then embed all skeletons with SBERT.
    """
    repo_file= SPIDER_DIR/"repo_cache.json"
    if repo_file.exists():
        d= json.load(open(repo_file,"r"))
        if "processed_count" not in d:
            d["processed_count"]=0
        if "items" not in d:
            d["items"]=[]
        if "simplified_map" not in d:
            d["simplified_map"]={}
    else:
        d= {"processed_count":0, "items":[], "simplified_map":{}}

    base= _load_json("train_spider")
    others= SPIDER_DIR/"train_others.json"
    if others.exists():
        base += json.load(open(others))

    max_size= limit if limit else len(base)
    start_idx= d["processed_count"]
    end_idx  = min(max_size, len(base))
    if start_idx>= end_idx:
        print(f"[build_repo] nothing to do. processed={start_idx} >= {end_idx}")
    else:
        print(f"[build_repo] from {start_idx} to {end_idx} of {len(base)}")
        for i in tqdm(range(start_idx,end_idx), desc="Simplify+Skeleton"):
            ex= base[i]
            q= ex["question"]
            sql= ex["query"]
            # if q not in d["simplified_map"], call deepseek once
            if q not in d["simplified_map"]:
                # call
                msgs= [
                    {"role":"system","content":"Rewrite questions in plain concise English."},
                    {"role":"user","content":f"Simplify: {q}"}
                ]
                # we do a short call
                try:
                    ans= call_deepseek(msgs, temperature=SIMPLIFY_TEMPERATURE, max_tokens=64)
                except Exception as e:
                    # if rate limit or 503, just fallback to original
                    es= str(e).lower()
                    if "429" in es or "503" in es or "rate limit" in es or "badgateway" in es:
                        ans= q
                    else:
                        raise
                d["simplified_map"][q]= ans
            q_simpl= d["simplified_map"][q]

            sk1= skeleton(q)
            sk2= skeleton(q_simpl)
            d["items"].append({"sk":sk1,"sql":sql,"nl":q})
            d["items"].append({"sk":sk2,"sql":sql,"nl":q})
        d["processed_count"]= end_idx
        json.dump(d, open(repo_file,"w"), indent=2)
        print(f"[build_repo] wrote {len(d['items'])} items total in {repo_file}")

    # embed
    all_sk= [x["sk"] for x in d["items"]]
    model= SentenceTransformer(EMBED_MODEL)
    arr= model.encode(all_sk, batch_size=64, show_progress_bar=True)
    arr/= np.linalg.norm(arr, axis=1, keepdims=True)
    np.save(EMBED_FILE, arr)
    print(f"[build_repo] saved {len(arr)} embeddings to {EMBED_FILE}")

def _load(split):
    return json.load(open(SPIDER_DIR/f"{split}.json"))
    
def load_repo():
    global model_sbert, train_pairs, train_vecs
    if train_vecs is not None: return
    repo_file= SPIDER_DIR/"repo_cache.json"
    if not repo_file.exists():
        raise ValueError("No repo_cache.json; run --build_repo first.")
    d= json.load(open(repo_file,"r"))
    items= d["items"]
    if not EMBED_FILE.exists():
        raise ValueError("No train_embeddings.npy found. run --build_repo first.")
    model_sbert= SentenceTransformer(EMBED_MODEL)
    arr= np.load(EMBED_FILE)
    if len(arr)!= len(items):
        raise ValueError(f"Inconsistent size: {len(arr)} vs {len(items)}")
    # store
    train_vecs= arr
    train_pairs=[]
    for it in items:
        train_pairs.append((it["sk"], it["sql"], it["nl"]))
    print(f"[load_repo] loaded {len(train_pairs)} skeleton->SQL items")

#####################################
# 7) retrieval
#####################################
def retrieve_dual(question:str, k1=2, k2=2):
    """Return a total of 4 example Q&A pairs from the repo: 2 from original skeleton, 2 from simplified skeleton.
       But for speed, we skip re-simplify in generation. We just do 2 from the same skeleton for both perspectives.
    """
    sk_o= skeleton(question)
    vec_o= model_sbert.encode([sk_o])[0]
    vec_o/= np.linalg.norm(vec_o)
    sims= train_vecs @ vec_o
    top_idxs= sims.argsort()[-(k1+k2):][::-1]  # total 4
    out=[]
    seen=set()
    for i in top_idxs:
        sk, sql, nl= train_pairs[i]
        pair= (nl, sql)
        if pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out

def build_demo_text(demo_items):
    lines=[]
    for (nl,sql) in demo_items:
        lines.append(f"Q: {nl}\nA: {sql};\n")
    return "\n".join(lines)

#####################################
# 8) postprocess & dynamic revision
#####################################
def postprocess_sql(ans:str):
    ans= re.sub(r"```+[\s\S]*?```+", "", ans)
    lines= ans.splitlines()
    keep=[]
    for ln in lines:
        if re.search(r"\bselect\b|\bupdate\b|\bdelete\b|\binsert\b|\bfrom\b|\bjoin\b|\bwhere\b|\bgroup by\b|\border by\b|;", ln.lower()):
            keep.append(ln)
    if not keep:
        keep= [ans]
    return " ".join(keep).strip()

def gen_sql(db_id, question):
    """2-turn dynamic revision, k1=2, k2=2, main calls with max_tokens=128, temp=0.3."""
    # retrieve demos
    demos= retrieve_dual(question, K1, K2)  # total 4
    demos_txt= build_demo_text(demos)
    sch= schema(db_id)
    prompt= f"""Schema: {sch}

Question: {question}

Here are example Q&A pairs:
{demos_txt}

Now output ONLY the final SQL for this question, with no extra text:
"""

    msgs= [
        {"role":"system","content":"You are a SQL generator. Output valid SQL only."},
        {"role":"user","content": prompt}
    ]

    final_sql= "SELECT 1"
    prev_res= None
    for attempt in range(MAX_TURNS):
        # prompt cache
        key= hashlib.sha256((MODEL_NAME + json.dumps(msgs)).encode()).hexdigest()
        ans= cache_get(key)
        if not ans:
            # do deepseek call
            try:
                ans= call_deepseek(msgs, temperature=MAIN_TEMPERATURE, max_tokens=128)
            except Exception as e:
                es= str(e).lower()
                if "429" in es or "badgateway" in es or "rate limit" in es or "503" in es:
                    # we skip or rotate
                    # for now just fallback to a second attempt
                    ans= "SELECT 1"
                else:
                    raise
            cache_put(key, ans)
        msgs.append({"role":"assistant","content":ans})

        candidate_sql= postprocess_sql(ans)
        try:
            r= execute(db_id, candidate_sql)
            if prev_res is not None and r==prev_res:
                return candidate_sql, r, True
            prev_res= r
            return candidate_sql, r, True
        except Exception as err:
            # second attempt => provide minimal feedback
            fb= f"Execution error: {err}"
            msgs.append({"role":"user","content":fb})

    return candidate_sql, [], False

#####################################
# 9) run & evaluate
#####################################
def run(split:str, limit=None):
    global DB_DIR
    if split == "test":
        DB_DIR = SPIDER_DIR / "test_database"
    else:
        DB_DIR = SPIDER_DIR / "database"

    data = _load(split)

    data= _load_json(split)
    if limit:
        data= data[:limit]
    preds=[]
    exact_cnt= 0
    exec_cnt= 0

    for ex in tqdm(data,desc=split):
        db_id= ex["db_id"]
        question= ex["question"]
        gold_sql= ex["query"]
        pred_sql, res, ok= gen_sql(db_id, question)
        preds.append(pred_sql)
        # exact
        if pred_sql.strip().lower()== gold_sql.strip().lower():
            exact_cnt+=1
        # exec
        if ok:
            gold_res= execute(db_id, gold_sql)
            if sorted(map(str,res))== sorted(map(str,gold_res)):
                exec_cnt+=1

    pred_path= ROOT/f"{split}.pred"
    pred_path.write_text("\n".join(preds))

    cmd= [
        sys.executable, EVAL_PY,
        "--gold", str(SPIDER_DIR/f"{split}_gold.sql"),
        "--pred", str(pred_path),
        "--db",   str(DB_DIR),
        "--table",str(TABLE_JSON),
        "--etype","all"
    ]
    out=subprocess.check_output(cmd, text=True)
    print(out)

    m= re.search(r"exact match accuracy: ([0-9.]+)", out.lower())
    em= float(m.group(1)) if m else 0.0
    s= re.search(r"test suite accuracy: ([0-9.]+)", out.lower())
    suite_acc= float(s.group(1)) if s else 0.0

    wandb.log({
        f"{split}/exact_string": exact_cnt/len(data),
        f"{split}/exec_match":   exec_cnt/len(data),
        f"{split}/exact_set":    em,
        f"{split}/test_suite":   suite_acc
    })

#####################################
# 10) CLI
#####################################
if __name__=="__main__":
    ap= argparse.ArgumentParser()
    ap.add_argument("--build_repo",action="store_true")
    ap.add_argument("--run_dev",   action="store_true")
    ap.add_argument("--run_test",  action="store_true")
    ap.add_argument("--limit",     type=int, default=None)
    args= ap.parse_args()

    wandb.init(entity=WB_ENTITY, project=WB_PROJECT,
               config={
                 "k1":K1,"k2":K2,
                 "max_turns":MAX_TURNS,
                 "main_temp":MAIN_TEMPERATURE,
                 "simplify_temp":SIMPLIFY_TEMPERATURE,
                 "model":"deepseek-r1"
               })

    if args.build_repo:
        build_repo(limit=args.limit)
    load_repo()

    if args.run_dev:
        run("dev", limit=args.limit)
    if args.run_test:
        run("test", limit=args.limit)

    wandb.finish()