#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Incremental retrieval-based Text-to-SQL pipeline
with single-provider "my_llm" usage,
and the same structure as the original code:

  - We no longer do round-robin or Groq calls.
  - If 429/502/503 or "bad gateway" error occurs, we fallback to a single retry or "SELECT 1".
  - Everything else is kept as close as possible to your original structure.
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
MAX_TURNS = 5  # dynamic revision attempts

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
print("LLM_API_KEY =", os.getenv("LLM_API_KEY"))
print("LLM_MODEL    =", os.getenv("LLM_MODEL"))
print("LLM_API_URL  =", os.getenv("LLM_API_URL"))

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
# 4) Single provider: "my_llm"
#####################################
PROVIDER = {
    "name":    "my_llm",
    "model":    os.getenv("LLM_MODEL", "gpt-4o-mini-2024-07-18"),
    "api_url":  os.getenv("LLM_API_URL", "https://api.nuwaapi.com/v1/chat/completions"),
    "api_key":  os.getenv("LLM_API_KEY",  "")
}

def call_my_llm(messages, temperature=1.0, max_tokens=256, top_p=1.0):
    """
    Single LLM call to 'my_llm', but with extra debug prints to show HTTP status
    and partial response text. If 429 or 503 => 'SELECT 1', if 502/gateway => retry once.
    """
    import requests
    import time

    headers = {
        "Authorization": f"Bearer {PROVIDER['api_key']}",
        "Content-Type":  "application/json"
    }
    data = {
        "model":       PROVIDER["model"],
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "top_p":       top_p
    }

    attempts = 2
    for i in range(attempts):
        try:
            resp = requests.post(PROVIDER["api_url"], headers=headers, json=data, timeout=60)
        except requests.exceptions.RequestException as e:
            # e.g. connection error => show error & retry once
            #print(f"[call_my_llm] Request error: {e}")
            if i == 0:
                time.sleep(1)
                continue
            else:
                return "SELECT 1"

        # Print debug info
        #print(f"[call_my_llm] Status: {resp.status_code}")
        #print(f"[call_my_llm] Response text snippet: {resp.text[:300]}...")

        if resp.status_code in [429, 503]:
            #print(f"[call_my_llm] {resp.status_code} => fallback SELECT 1")
            return "SELECT 1"

        lower_txt = resp.text.lower()
        if resp.status_code == 502 or "gateway" in lower_txt or "bad gateway" in lower_txt:
            if i == 0:
                print("[call_my_llm] 502/gateway => sleep & retry once")
                time.sleep(1)
                continue
            else:
                print("[call_my_llm] 502/gateway again => fallback SELECT 1")
                return "SELECT 1"

        if resp.status_code != 200:
            # Some other error
            raise RuntimeError(f"[call_my_llm] LLM error {resp.status_code}: {resp.text}")

        # If success => parse JSON
        js = resp.json()
        answer_str = js["choices"][0]["message"]["content"].strip()
        #print(f"[call_my_llm] Raw LLM answer: {answer_str[:200]}...")
        return answer_str

    # If all attempts exhausted
    return "SELECT 1"
    """
    Single LLM call to 'my_llm' with minimal fallback for 429/502/503 or 'gateway' error:
     - If 429 or 503 => fallback "SELECT 1"
     - If 502 or 'gateway' => do 1 sleep + 1 retry
     - Otherwise raise on non-200 status.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {PROVIDER['api_key']}",
        "Content-Type":  "application/json"
    }
    data = {
        "model":       PROVIDER["model"],
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "top_p":       top_p
    }

    # We'll do up to 2 tries if 502/gateway
    attempts = 2
    for i in range(attempts):
        try:
            resp = requests.post(PROVIDER["api_url"], headers=headers, json=data, timeout=60)
        except requests.exceptions.RequestException as e:
            # connection error => retry once
            if i == 0:
                time.sleep(1)
                continue
            else:
                return "SELECT 1"

        if resp.status_code in [429, 503]:
            # immediate fallback
            return "SELECT 1"

        txt_lower = resp.text.lower()
        if resp.status_code == 502 or "gateway" in txt_lower or "bad gateway" in txt_lower:
            if i == 0:
                time.sleep(1)
                continue
            else:
                return "SELECT 1"

        if resp.status_code != 200:
            # some other failure
            raise RuntimeError(f"LLM error {resp.status_code}: {resp.text}")

        js = resp.json()
        return js["choices"][0]["message"]["content"].strip()

    # if all attempts fail
    return "SELECT 1"

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

#_tok_pat = re.compile(r"[A-Za-z_]+$")
_SCHEMA_TOKENS = set()
def init_schema_tokens():
    """
    Parse 'tables.json' or spider schema to gather all table/column names.
    Add them in lowercase to _SCHEMA_TOKENS.
    """
    global _SCHEMA_TOKENS

    table_json_path = SPIDER_DIR / "tables.json"
    if not table_json_path.exists():
        print(f"Warning: {table_json_path} not found. _SCHEMA_TOKENS stays empty.")
        return

    # 'tables.json' is a list of { "db_id":..., "table_names_original":..., "column_names_original":... } etc.
    data = json.load(open(table_json_path, "r"))

    schema_tokens = set()
    for db_info in data:
        # table_names_original might look like ["city", "country", ...]
        for tname in db_info.get("table_names_original", []):
            schema_tokens.add(tname.lower())

        # column_names_original is a list of [ (table_idx, "column_name"), ...]
        for col in db_info.get("column_names_original", []):
            col_name = col[1]
            schema_tokens.add(col_name.lower())

    # optionally add table aliases if you use them, or other known keywords
    _SCHEMA_TOKENS = schema_tokens
    print(f"[init_schema_tokens] Loaded {len(_SCHEMA_TOKENS)} schema tokens.")

def skeleton(text: str):
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    sk = []
    for t in tokens:
        low_t = t.lower()
        if low_t in _SCHEMA_TOKENS:  
            sk.append("<SCHEMA>")
        elif t.isnumeric():
            sk.append("<NUM>")
        else:
            sk.append(low_t)
    return " ".join(sk)

    #return " ".join(w.lower() for w in text.split() if _tok_pat.match(w))

#####################################
# 7) simplify with caching
#####################################
def simplify_question(q: str):
    """Use single 'my_llm' provider with SIMPLIFY_TEMPERATURE=1.0, but memoize results."""
    if q in simplify_cache:
        return simplify_cache[q]

    msgs = [
        {"role":"system","content":"Replace the words as far as possible to simplify the question, "
        "making it syntactically clear, common and easy to understand."},
        {"role":"user",   "content":f"Simplify: {q}"}
    ]
    try:
        ans = call_my_llm(msgs, temperature=SIMPLIFY_TEMPERATURE, max_tokens=64, top_p=1.0)
    except Exception as e:
        # fallback
        ans = q
    simplify_cache[q] = ans
    return ans

#####################################
# 8) Build repo with "nl"
#####################################
def build_repo(limit=None):
    global model_sbert, train_pairs, train_vecs

    # 1) read the schema tokens first
    init_schema_tokens()

    repo_cache_path = SPIDER_DIR / "repo_cache.json"
    if repo_cache_path.exists():
        with open(repo_cache_path, "r") as f:
            repo_data = json.load(f)
    else:
        repo_data = {"processed_count": 0, "items": []}

    base = _load("train_spider")
    others_file = SPIDER_DIR / "train_others.json"
    if others_file.exists():
        base += json.load(open(others_file))

    max_size  = limit if limit else len(base)
    start_idx = repo_data["processed_count"]
    end_idx   = min(max_size, len(base))
    if start_idx >= end_idx:
        print(f"[build_repo] Nothing to do: processed_count={start_idx}, end_idx={end_idx}")
        return
    else:
        print(f"[build_repo] from {start_idx} to {end_idx} of {len(base)}")
        new_items=[]
        from tqdm import tqdm
        for i in tqdm(range(start_idx, end_idx), desc="Simplify+Skeleton"):
            ex = base[i]
            q_orig = ex["question"]
            sql    = ex["query"]
            q_simpl= simplify_question(q_orig)

            sk1 = skeleton(q_orig)
            sk2 = skeleton(q_simpl)

            new_items.append({"sk": sk1, "sql": sql, "nl": q_orig})
            new_items.append({"sk": sk2, "sql": sql, "nl": q_orig})

        repo_data["items"].extend(new_items)
        repo_data["processed_count"] = end_idx
        with open(repo_cache_path,"w") as f:
            json.dump(repo_data, f, indent=2)
        print(f"[build_repo] wrote repo_cache.json with total {len(repo_data['items'])} items")

    # 2) Now embed all skeletons
    all_sk = [x["sk"] for x in repo_data["items"]]
    from sentence_transformers import SentenceTransformer
    model_sbert = SentenceTransformer(EMBED_MODEL)
    vecs = model_sbert.encode(all_sk, batch_size=64, show_progress_bar=True)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(EMBED_FILE, vecs)
    print(f"[build_repo] saved {len(vecs)} embeddings to {EMBED_FILE}")

def load_repo():
    global model_sbert, train_pairs, train_vecs
    if train_vecs is not None:
        return
    repo_file = SPIDER_DIR / "repo_cache.json"
    if not repo_file.exists():
        raise FileNotFoundError("No repo_cache.json; run --build_repo first.")
    d = json.load(open(repo_file,"r"))
    items = d["items"]
    emb_file = EMBED_FILE
    if not emb_file.exists():
        raise FileNotFoundError(f"No {emb_file}; run --build_repo first.")

    from sentence_transformers import SentenceTransformer
    model_sbert = SentenceTransformer(EMBED_MODEL)
    arr = np.load(emb_file)
    if len(arr) != len(items):
        raise ValueError(f"Inconsistent size {len(arr)} vs {len(items)}")
    # store
    train_pairs = [(it["sk"], it["sql"], it["nl"]) for it in items]
    train_vecs  = arr
    print(f"[load_repo] loaded {len(train_pairs)} skeleton->(sql,nl) items")

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
        short = ", ".join(cols[:10]) + (", …" if len(cols)>10 else "")
        out.append(f"{tb}(" + short +")")
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
    # original
    orig_sk= skeleton(q)
    q_simpl= simplify_question(q)
    simp_sk= skeleton(q_simpl)

    vec_o= model_sbert.encode([orig_sk])[0]; vec_o/= np.linalg.norm(vec_o)
    sims_o= train_vecs @ vec_o
    top_o= sims_o.argsort()[-k1:][::-1]

    vec_s= model_sbert.encode([simp_sk])[0]; vec_s/= np.linalg.norm(vec_s)
    sims_s= train_vecs @ vec_s
    top_s= sims_s.argsort()[-k2:][::-1]

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
    blocks=[]
    for (nl, sql) in demo_items:
        blocks.append(f"Q: {nl}\nA: {sql};")
    return "\n\n".join(blocks)

#####################################
# 12) postprocess / explain
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

def explain_sql(sql, question):
    """
    We'll do a quick explanation call if needed.
    """
    # msgs= [
    #     {"role":"system","content":"You explain SQL in plain English."},
    #     {"role":"user","content":f"Explain: {sql}"}
    # ]
    msgs = [
    {"role":"system","content":
        "You are an analyst. First paraphrase the SQL in plain English. "
        "Then list the semantic differences between that paraphrase and the QUESTION below."},
    {"role":"user","content":f"SQL:\n{sql}\n\nQUESTION:\n{question}"}
    ]   

    try:
        return call_my_llm(msgs, max_tokens=128, temperature=MAIN_TEMPERATURE, top_p=1.0)
    except:
        return "No explanation"

#####################################
# 13) gen_sql with dynamic revision
#####################################
def gen_sql(db_id, q):
    """
    1) retrieve demos
    2) build prompt
    3) attempt up to MAX_TURNS
       - if success => return
       - else => append error feedback => next attempt
    """
    demo_items= retrieve_dual(q, K1, K2)
    demos_text= build_demo_text(demo_items)

    prompt= f"""Schema: {schema(db_id)}

Question: {q}

Here are example Q&A pairs:
{demos_text}

Now output ONLY the final SQL for this question, with no extra text:
"""

    msgs= [
        {"role":"system","content":"You are a SQL generator. Output valid SQL only, always use table.column format to avoid ambiguity."},
        {"role":"user","content": prompt}
    ]

    prev_res= None
    final_sql= "SELECT 1"

    for attempt in range(MAX_TURNS):
        # prompt cache
        hkey = hashlib.sha256(("my_llm"+json.dumps(msgs)).encode()).hexdigest()
        ans  = cache_get(hkey)
        if not ans:
            try:
                ans= call_my_llm(msgs, temperature=MAIN_TEMPERATURE, max_tokens=256, top_p=1.0)
            except Exception as e:
                # if we see 429/503 => SELECT 1 or similar
                # else re-raise
                if any(x in str(e).lower() for x in ["429","rate limit","503","service unavailable"]):
                    ans= "SELECT 1"
                else:
                    raise
            cache_put(hkey, ans)

        msgs.append({"role":"assistant","content":ans})
        final_sql= postprocess_sql(ans)

        try:
            res = execute(db_id, final_sql)
            # if repeated => end
            if prev_res is not None and res==prev_res:
                return final_sql, res, True
            prev_res= res
            return final_sql, res, True

        except Exception as err:
            # add feedback
            explanation= explain_sql(final_sql, q)
            snippet= short_preview(db_id, final_sql)
            feedback= [
                f"Execution error: {err}",
                f"NL Explanation: {explanation}",
                f"List mismatches between this SQL and the question: {q}",
                f"Relevant table snippet:\n{snippet}"
            ]
            msgs.append({"role":"user","content":"\n".join(feedback)})

    return final_sql, [], False

#####################################
# 14) run & evaluate
#####################################
def run(split, limit=None):
    global DB_DIR
    if split == "test":
        DB_DIR = SPIDER_DIR / "test_database"
    else:
        DB_DIR = SPIDER_DIR / "database"

    data = _load(split)

    data = _load(split)
    if limit:
        data= data[:limit]
    preds=[]
    exact_cnt=0
    exec_cnt=0

    for ex in tqdm(data,desc=split):
        db_id= ex["db_id"]
        question= ex["question"]
        gold_sql= ex["query"]

        sql_pred, res, ok= gen_sql(db_id, question)
        preds.append(sql_pred)

        # exact string
        if sql_pred.strip().lower() == gold_sql.strip().lower():
            exact_cnt+=1
        # exec
        if ok:
            gold_res= execute(db_id, gold_sql)
            if equal(res,gold_res):
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
    out= subprocess.check_output(cmd, text=True)
    print(out)

    # parse "exact match accuracy: x"
    match= re.search(r"exact match accuracy: ([0-9.]+)", out.lower())
    em= float(match.group(1)) if match else 0.0

    # parse "test suite accuracy: x"
    suite= re.search(r"test suite accuracy: ([0-9.]+)", out.lower())
    suite_acc= float(suite.group(1)) if suite else 0.0

    wandb.log({
        f"{split}/exact_string": exact_cnt/len(data),
        f"{split}/exec_match":   exec_cnt/len(data),
        f"{split}/exact_set":    em,
        f"{split}/test_suite":   suite_acc
    })

#####################################
# 15) CLI
#####################################
if __name__=="__main__":
    ap= argparse.ArgumentParser()
    ap.add_argument("--build_repo",action="store_true")
    ap.add_argument("--run_dev",   action="store_true")
    ap.add_argument("--run_test",  action="store_true")
    ap.add_argument("--limit",     type=int, default=None)
    args= ap.parse_args()

    wandb.init(entity=WB_ENTITY, project=WB_PROJECT)
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
