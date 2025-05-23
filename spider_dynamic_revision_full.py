#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Incremental retrieval-based Text-to-SQL pipeline
with dual-perspective retrieval, question+SQL demos, semantic-diff feedback,
shortened table preview, early stopping on repeated result,
and new optimizations:
  1) separate temperature for simplify vs main generation
  2) parse "test suite accuracy"
  3) deduplicate demos
  4) cache simplify calls
  5) handle BadGateway/502
"""

import os, sys, json, sqlite3, time, argparse, subprocess, hashlib, re, random
from collections import deque
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import wandb, dotenv

#####################################
# 0) Additional hyperparams
#####################################
MAIN_TEMPERATURE = 0.5
SIMPLIFY_TEMPERATURE = 1.0

K1, K2 = 4, 4  # retrieval counts from original vs simplified skeleton

MAX_TURNS = 3  # dynamic revision attempts

simplify_cache = {}  # to memoize "Simplify: question" calls

#####################################
# 1) NLTK resources (if needed)
#####################################
try:
    import nltk
    nltk.download('punkt', quiet=True)
except:
    pass

#####################################
# 2) Load environment variables
#####################################
dotenv.load_dotenv(override=True)
print("GROQ_API_KEY_L3A =", os.getenv("GROQ_API_KEY_L3A"))
print("GROQ_API_KEY_M7B =", os.getenv("GROQ_API_KEY_M7B"))
print("GROQ_API_KEY_L3B =", os.getenv("GROQ_API_KEY_L3B"))

#####################################
# 3) Paths & constants
#####################################
ROOT        = Path(__file__).resolve().parent
SPIDER_DIR  = Path(os.getenv("SPIDER_DIR", "~/Big_data/spider_data")).expanduser()
EVAL_PY     = "spider/evaluation.py"
#DB_DIR      = SPIDER_DIR / "database"
TABLE_JSON  = SPIDER_DIR / "tables.json"
EMBED_FILE  = SPIDER_DIR / "train_embeddings.npy"
CACHE_DB    = ROOT / "prompt_cache.sqlite"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

WB_ENTITY   = "saint-yangyuqi-university-of-z-rich"
WB_PROJECT  = "text2sql-project"

#####################################
# 4) Multi-provider round-robin
#####################################
PROVIDERS = deque([
    dict(name="groq", model="llama-3.1-8b-instant", key_env="GROQ_API_KEY_L3A"),
    dict(name="groq", model="gemma2-9b-it",         key_env="GROQ_API_KEY_M7B"),
    dict(name="groq", model="llama3-8b-8192",       key_env="GROQ_API_KEY_L3B"),
])

def _next_provider():
    """Round-robin among providers."""
    p = PROVIDERS[0]
    PROVIDERS.rotate(-1)
    return p

def _client(provider):
    """Return a function that does chat -> str. We'll pass temperature in calls."""
    if provider["name"] == "groq":
        from groq import Groq
        cli = Groq(api_key=os.getenv(provider["key_env"], ""))
        return lambda msgs, **kw: cli.chat.completions.create(
            model=provider["model"], messages=msgs, **kw
        ).choices[0].message.content.strip()
    else:
        raise ValueError("Unsupported provider:", provider)

#####################################
# 5) Prompt cache
#####################################
cache_conn = sqlite3.connect(CACHE_DB)
cache_conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
cache_conn.commit()

def cache_get(h):
    row = cache_conn.execute("SELECT v FROM cache WHERE k=?", (h,)).fetchone()
    return None if row is None else row[0]

def cache_put(h, v):
    cache_conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (h, v))
    cache_conn.commit()

#####################################
# 6) SBERT-based retrieval
#####################################
model_sbert, train_pairs, train_vecs = None, None, None

def _load(split):
    return json.load(open(SPIDER_DIR/f"{split}.json"))

_tok_pat = re.compile(r"[A-Za-z_]+$")

def skeleton(text: str):
    return " ".join(w.lower() for w in text.split() if _tok_pat.match(w))

#####################################
# 7) simplify with caching
#####################################
def simplify_question(q: str):
    """Use round-robin LLM with SIMPLIFY_TEMPERATURE=1.0, but memoize results."""
    if q in simplify_cache:
        return simplify_cache[q]

    p = _next_provider()
    chat = _client(p)
    msgs = [
        {"role":"system","content":"Rewrite questions in plain concise English."},
        {"role":"user",   "content":f"Simplify: {q}"}
    ]
    # We'll do an ephemeral call with temperature=SIMPLIFY_TEMPERATURE
    # and maybe top_p=1.0 to let it be freer:
    try:
        ans = chat(msgs, max_tokens=64, temperature=SIMPLIFY_TEMPERATURE, top_p=1.0)
    except Exception as e:
        # If we see "429", "rate limit", "BadGateway", "502", rotate provider
        #if ("429" in str(e)) or ("rate limit" in str(e).lower()) or ("502" in str(e)) or ("badgateway" in str(e).lower()):
        if ("429" in str(e)) or ("rate limit" in str(e).lower())or ("502" in str(e)) or ("badgateway" in str(e).lower()) or ("503" in str(e)) or ("service unavailable" in str(e).lower()):
            # we can do one retry:
            p = _next_provider()
            chat = _client(p)
            ans = chat(msgs, max_tokens=64, temperature=SIMPLIFY_TEMPERATURE, top_p=1.0)
        else:
            raise
    simplify_cache[q] = ans
    return ans

#####################################
# 8) Build repo with "nl"
#####################################
def build_repo(limit=None):
    global model_sbert, train_pairs, train_vecs

    repo_cache_path = SPIDER_DIR / "repo_cache.json"
    if repo_cache_path.exists():
        with open(repo_cache_path,"r") as f:
            repo_data = json.load(f)
    else:
        repo_data = {"processed_count":0,"items":[]}

    base = _load("train_spider")
    others_file = SPIDER_DIR/"train_others.json"
    if others_file.exists():
        base += json.load(open(others_file))

    max_size = limit if limit else len(base)
    start_idx= repo_data["processed_count"]
    end_idx  = min(max_size, len(base))
    if start_idx>=end_idx:
        print(f"[build_repo] Nothing to do: processed_count={start_idx}, end_idx={end_idx}")
    else:
        print(f"[build_repo] from {start_idx} to {end_idx} (of {len(base)})")
        new_items=[]
        for i in tqdm(range(start_idx,end_idx), desc="Simplify+Skeleton"):
            ex = base[i]
            q_orig = ex["question"]
            sql    = ex["query"]
            q_simpl= simplify_question(q_orig)

            sk1 = skeleton(q_orig)
            sk2 = skeleton(q_simpl)

            # store them
            new_items.append({"sk":sk1,"sql":sql,"nl":q_orig})
            new_items.append({"sk":sk2,"sql":sql,"nl":q_orig})

        repo_data["items"].extend(new_items)
        repo_data["processed_count"] = end_idx
        with open(repo_cache_path,"w") as f:
            json.dump(repo_data,f,indent=2)
        print(f"[build_repo] wrote repo_cache.json with total {len(repo_data['items'])} items")

    # Now embed
    all_sk = [x["sk"] for x in repo_data["items"]]
    model_sbert = SentenceTransformer(EMBED_MODEL)
    vecs = model_sbert.encode(all_sk, batch_size=64, show_progress_bar=True)
    vecs/= np.linalg.norm(vecs,axis=1,keepdims=True)
    np.save(EMBED_FILE, vecs)
    print(f"[build_repo] saved {len(vecs)} embeddings to {EMBED_FILE}")

def load_repo():
    global model_sbert, train_pairs, train_vecs
    if train_vecs is not None:
        return
    repo_cache_path = SPIDER_DIR / "repo_cache.json"
    if not repo_cache_path.exists():
        raise ValueError("No 'repo_cache.json' found! run --build_repo first.")

    with open(repo_cache_path,"r") as f:
        repo_data = json.load(f)
    items = repo_data["items"]

    emb_file = EMBED_FILE
    if not emb_file.exists():
        raise ValueError(f"No {emb_file} found! run --build_repo first.")

    model_sbert = SentenceTransformer(EMBED_MODEL)
    arr = np.load(emb_file)
    if len(arr)!= len(items):
        raise ValueError(f"Inconsistent size {len(arr)} vs {len(items)}")

    train_pairs = [(it["sk"],it["sql"],it["nl"]) for it in items]
    train_vecs  = arr
    print(f"[load_repo] loaded {len(train_pairs)} skeleton->(sql,nl)")

#####################################
# 9) DB helpers
#####################################
_db = {}
def conn(db_id):
    if db_id not in _db:
        path = DB_DIR / db_id / f"{db_id}.sqlite"
        c = sqlite3.connect(path)
        c.text_factory = lambda b: b.decode(errors="replace")
        _db[db_id] = c
    return _db[db_id]

def schema(db_id):
    c = conn(db_id).cursor()
    t= [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out=[]
    for tb in t:
        if tb.startswith("sqlite"):
            continue
        cols= [col[1] for col in c.execute(f"PRAGMA table_info('{tb}')")]
        out.append(f"{tb}(" + ", ".join(cols[:10]) + (", …" if len(cols)>10 else "") +")")
    return " ; ".join(out)

def execute(db_id, sql):
    c = conn(db_id).cursor()
    c.execute(sql)
    return c.fetchall()

def equal(a, b):
    return sorted(map(str,a))==sorted(map(str,b))

#####################################
# 10) Table snippet
#####################################
def relevant_columns(sql):
    # naive search
    matches = re.findall(r"(?:select|where|group by|order by)\s+([\w, *]+)", sql, flags=re.I)
    colset=[]
    for m in matches:
        colset.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m))
    return list(set(colset))

def short_preview(db_id, sql):
    cols = relevant_columns(sql)
    tbls= re.findall(r"from\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.I)
    tbls= list(set(tbls))
    if not tbls:
        return "No table in query"
    out=[]
    c= conn(db_id).cursor()
    for t in tbls:
        try:
            col_info = [col[1] for col in c.execute(f"PRAGMA table_info('{t}')")]
            used_cols= [co for co in cols if co in col_info]
            if not used_cols:
                used_cols=["*"]
            q = f"SELECT {', '.join(used_cols)} FROM {t} LIMIT 1"
            rows= c.execute(q).fetchall()
            out.append(f"Table {t}, columns {used_cols}, sample row: {rows}")
        except:
            pass
    if not out:
        return "No snippet available"
    return "\n".join(out)

#####################################
# 11) Dual retrieval + de-dupe
#####################################
def retrieve_dual(q, k1, k2):
    """
    from original skeleton and from simplified skeleton
    build final list of (nl,sql), removing duplicates
    """
    # original
    orig_sk= skeleton(q)
    # simplified
    q_simpl= simplify_question(q)
    simp_sk= skeleton(q_simpl)

    vec_o= model_sbert.encode([orig_sk])[0]; vec_o/= np.linalg.norm(vec_o)
    sims_o= train_vecs@vec_o
    top_o= sims_o.argsort()[-k1:][::-1]

    vec_s= model_sbert.encode([simp_sk])[0]; vec_s/= np.linalg.norm(vec_s)
    sims_s= train_vecs@vec_s
    top_s= sims_s.argsort()[-k2:][::-1]

    # gather
    items=[]
    seen=set()
    for i in top_o:
        sk, sql, nl = train_pairs[i]
        if (nl, sql) not in seen:
            items.append((nl,sql))
            seen.add((nl,sql))
    for j in top_s:
        sk, sql, nl = train_pairs[j]
        if (nl, sql) not in seen:
            items.append((nl,sql))
            seen.add((nl,sql))
    return items

def build_demo_text(demo_items):
    # each item is (nl, sql)
    blocks=[]
    for (nl, sql) in demo_items:
        # add "Q: …" / "A: …;"
        blocks.append(f"Q: {nl}\nA: {sql};")
    return "\n\n".join(blocks)

#####################################
# postprocess / explain
#####################################
def postprocess_sql(ans:str):
    ans= re.sub(r"```+[\s\S]*?```+", "", ans)
    lines= ans.splitlines()
    keep=[]
    for ln in lines:
        if re.search(r"\bselect\b|\bupdate\b|\bdelete\b|\binsert\b|\bfrom\b|\bjoin\b|\bwhere\b|\bgroup by\b|\border by\b|;", ln.lower()):
            keep.append(ln)
    if not keep:
        keep=[ans]
    return " ".join(keep).strip()

def explain_sql(sql, chat):
    msgs= [
        {"role":"system","content":"You explain SQL in plain English."},
        {"role":"user","content":f"Explain: {sql}"}
    ]
    try:
        return chat(msgs, max_tokens=128, temperature=MAIN_TEMPERATURE, top_p=1.0)
    except Exception as e:
        if ("429" in str(e)) or ("rate limit" in str(e).lower()) or ("502" in str(e)) or ("badgateway" in str(e).lower()):
            # rotate
            p = _next_provider()
            chat2= _client(p)
            return chat2(msgs, max_tokens=128, temperature=MAIN_TEMPERATURE, top_p=1.0)
        else:
            raise

#####################################
# 12) gen_sql with dynamic revision
#####################################
def gen_sql(db_id, q):
    p = _next_provider()
    chat = _client(p)

    # retrieve 8 demos total
    demo_items= retrieve_dual(q, K1, K2)
    demos_text= build_demo_text(demo_items)

    prompt= f"""Schema: {schema(db_id)}

Question: {q}

Here are example Q&A pairs:
{demos_text}

Now output ONLY the final SQL for this question, with no extra text:
"""

    msgs=[
        {"role":"system","content":"You are a SQL generator. Output valid SQL only."},
        {"role":"user","content": prompt}
    ]

    prev_res= None
    final_sql="SELECT 1"
    for attempt in range(MAX_TURNS):
        # prompt cache
        hkey= hashlib.sha256((p["model"]+json.dumps(msgs)).encode()).hexdigest()
        ans= cache_get(hkey)
        if not ans:
            try:
                # main generation with temperature=MAIN_TEMPERATURE
                ans= chat(msgs, max_tokens=256, temperature=MAIN_TEMPERATURE, top_p=1.0)
            except Exception as e:
                #if ("429" in str(e)) or ("rate limit" in str(e).lower()) or ("502" in str(e)) or ("badgateway" in str(e).lower()):
                if ("429" in str(e)) or ("rate limit" in str(e).lower()) or ("502" in str(e)) or ("badgateway" in str(e).lower()) or ("503" in str(e)) or ("service unavailable" in str(e).lower()):
                    p= _next_provider()
                    chat= _client(p)
                    # re-try once
                    ans= chat(msgs, max_tokens=256, temperature=MAIN_TEMPERATURE, top_p=1.0)
                else:
                    raise
            cache_put(hkey, ans)
        msgs.append({"role":"assistant","content":ans})

        final_sql= postprocess_sql(ans)
        try:
            res = execute(db_id, final_sql)
            # success
            if prev_res is not None and res==prev_res:
                # same result => no improvement => end
                return final_sql, res, True
            prev_res= res
            return final_sql, res, True
        except Exception as err:
            # feedback
            feedback=[]
            feedback.append(f"Execution error: {err}")
            explanation= explain_sql(final_sql, chat)
            feedback.append(f"NL Explanation: {explanation}")
            feedback.append(f"List mismatches between this SQL and the original question: {q}")
            msgs.append({"role":"user","content":"\n".join(feedback)})
            snippet= short_preview(db_id, final_sql)
            msgs.append({"role":"user","content":f"Relevant table snippet:\n{snippet}"})

    return final_sql, [], False

#####################################
# 13) run & evaluate
#####################################
def run(split, limit=None):
    global DB_DIR
    if split == "test":
        DB_DIR = SPIDER_DIR / "test_database"
    else:
        DB_DIR = SPIDER_DIR / "database"
    data= _load(split)
    if limit:
        data= data[:limit]
    preds=[]
    exact_cnt=0
    exec_cnt=0

    for ex in tqdm(data,desc=split):
        sql_pred, res, ok= gen_sql(ex["db_id"], ex["question"])
        preds.append(sql_pred)

        # exact string
        if sql_pred.strip().lower()== ex["query"].strip().lower():
            exact_cnt+=1
        # exec
        if ok:
            gold_res= execute(ex["db_id"], ex["query"])
            if equal(res,gold_res):
                exec_cnt+=1

    pred_path= ROOT/f"{split}.pred"
    pred_path.write_text("\n".join(preds))

    cmd=[
        sys.executable,
        EVAL_PY,
        "--gold", str(SPIDER_DIR/f"{split}_gold.sql"),
        "--pred", str(pred_path),
        "--db",   str(DB_DIR),
        "--table",str(TABLE_JSON),
        "--etype","all"
    ]
    out=subprocess.check_output(cmd, text=True)
    print(out)

    # parse "exact match accuracy: x"
    match= re.search(r"exact match accuracy: ([0-9.]+)", out.lower())
    em= float(match.group(1)) if match else 0.0

    # parse "test suite accuracy: x" if present
    suite= re.search(r"test suite accuracy: ([0-9.]+)", out.lower())
    suite_acc= float(suite.group(1)) if suite else 0.0

    wandb.log({
        f"{split}/exact_string": exact_cnt/len(data),
        f"{split}/exec_match":   exec_cnt/len(data),
        f"{split}/exact_set":    em,
        f"{split}/test_suite":   suite_acc
    })

#####################################
# 14) CLI
#####################################
if __name__=="__main__":
    ap= argparse.ArgumentParser()
    ap.add_argument("--build_repo",action="store_true")
    ap.add_argument("--run_dev",   action="store_true")
    ap.add_argument("--run_test",  action="store_true")
    ap.add_argument("--limit",     type=int, default=None)
    args= ap.parse_args()

    wandb.init(entity=WB_ENTITY, project=WB_PROJECT)
    # Log these hyperparams to W&B so we know what config produced each run
    wandb.config.update({
        "max_turns":   MAX_TURNS,
        "k1":          K1,
        "k2":          K2,
        "temp_main":   MAIN_TEMPERATURE,
        "temp_simplify": SIMPLIFY_TEMPERATURE
    })

    if args.build_repo:
        build_repo(limit=args.limit)
    load_repo()
    if args.run_dev:
        run("dev", args.limit)
    if args.run_test:
        run("test", args.limit)

    wandb.finish()
