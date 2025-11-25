# agent.py — Updated: prefer qwen/qwen3-32b, robust JSON parsing, metric fallback
import os
import json
import re
from datetime import datetime, timedelta
from math import ceil
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"

# -------------------------
# Load helpers & config
# -------------------------
def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Could not load {path}: {e}")
        return {}

METADATA = load_json_file("metadata.json")   # optional column descriptions
METRICS = load_json_file("metrics.json") or {}

# build a synonyms -> metric key map for quick lookup
SYNONYM_MAP = {}
for k, v in METRICS.items():
    syns = v.get("synonyms", []) if isinstance(v, dict) else []
    # include metric key itself as synonym
    all_syns = set([k.lower()] + [s.lower() for s in syns])
    for s in all_syns:
        SYNONYM_MAP[s] = k

# -------------------------
# SQL Schema
# -------------------------
def load_sql_schema():
    try:
        import pyodbc
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            f"Encrypt=no;TrustServerCertificate=yes;",
            timeout=5
        )
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{TABLE_NAME}'
        """)
        schema = {row.COLUMN_NAME: row.DATA_TYPE.lower() for row in cursor.fetchall()}
        conn.close()
        return schema
    except Exception as e:
        print("⚠ Could not load schema:", e)
        return {}

_SCHEMA = {}
def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA

# -------------------------
# GROQ client & auto-model
# -------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        print("⚠ Could not create GROQ client:", e)
        return None

def choose_best_groq_model(client):
    """
    Prefer qwen/qwen3-32b as you requested, then fallbacks.
    """
    preferred = [
        "qwen/qwen3-32b",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant"
    ]
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        for p in preferred:
            if p in available:
                return p
        # otherwise choose first 'qwen' or 'llama' or fallback to first available
        for m in available:
            if "qwen" in m or "llama" in m:
                return m
        return available[0] if available else None
    except Exception as e:
        # if listing fails, fallback to qwen name (best effort)
        print("⚠ Could not list Groq models:", e)
        return "qwen/qwen3-32b"

# -------------------------
# Time utilities (FY starts in MARCH)
# -------------------------
def calendar_to_fy_year(year:int, month:int) -> int:
    return year if month >= 3 else year - 1

def month_name_to_num(name: str):
    try:
        return datetime.strptime(name[:3].capitalize(), "%b").month
    except Exception:
        try:
            return datetime.strptime(name, "%B").month
        except Exception:
            return None

def current_utc():
    return datetime.utcnow()

def calendar_to_fy_month(calendar_month:int) -> int:
    return ((calendar_month - 3) % 12) + 1

def fy_quarter_from_fy_month(fy_month:int) -> int:
    return ceil(fy_month / 3)

def last_financial_quarter(reference=None):
    if not reference:
        reference = current_utc()
    fy_year = calendar_to_fy_year(reference.year, reference.month)
    fy_month = calendar_to_fy_month(reference.month)
    fq = fy_quarter_from_fy_month(fy_month)
    if fq == 1:
        return fy_year - 1, 4
    else:
        return fy_year, fq - 1

def previous_calendar_month(reference=None):
    if not reference:
        reference = current_utc()
    first = reference.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.year, prev.month

def previous_calendar_year(reference=None):
    if not reference:
        reference = current_utc()
    return reference.year - 1

# -------------------------
# Helpers
# -------------------------
def sanitize_filter_value(val):
    if isinstance(val, dict):
        for k in ("year","quarter","month","value"):
            if k in val:
                return val[k]
        for v in val.values():
            if isinstance(v, int):
                return v
        return str(val)
    if isinstance(val, list):
        return val
    return val

# -------------------------
# Time phrase parser
# -------------------------
def parse_time_filters(text: str):
    q = (text or "").lower()
    filters = []

    m = re.search(r'\b(?:fy|financial year)\s*[:#-]?\s*(20\d{2})\b', q)
    if m:
        fy = int(m.group(1))
        col = "FinancialYear"
        filters.append({"column": col, "operator":"=", "value": fy})
        return filters

    m = re.search(r'\b(?:q|quarter)\s*[-:\s]*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        qnum = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else None
        if year:
            filters.append({"column":"FinancialQuarter","operator":"=","value": qnum})
            filters.append({"column":"FinancialYear","operator":"=","value": year})
        else:
            filters.append({"column":"FinancialQuarter","operator":"=","value": qnum})
        return filters

    m = re.search(r'\b('
                  r'january|february|march|april|may|june|july|august|september|october|november|december|'
                  r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
                  r')\s+(20\d{2})\b', q, flags=re.IGNORECASE)
    if m:
        mon = m.group(1)
        yr = int(m.group(2))
        cal_month = month_name_to_num(mon)
        if cal_month:
            fy_month = calendar_to_fy_month(cal_month)
            fy_year = calendar_to_fy_year(yr, cal_month)
            filters.append({"column":"FinancialMonth","operator":"=","value": fy_month})
            filters.append({"column":"FinancialYear","operator":"=","value": fy_year})
            return filters

    if re.search(r'\blast month\b|\bprevious month\b', q):
        y, m = previous_calendar_month()
        fy = calendar_to_fy_year(y, m)
        fm = calendar_to_fy_month(m)
        filters.append({"column":"FinancialMonth","operator":"=","value": fm})
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    if re.search(r'\blast quarter\b|\bprevious quarter\b', q):
        fy, fq = last_financial_quarter()
        filters.append({"column":"FinancialQuarter","operator":"=","value": fq})
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    if re.search(r'\blast year\b|\bprevious year\b', q):
        prev = previous_calendar_year()
        filters.append({"column":"FinancialYear","operator":"=","value": prev})
        return filters

    m = re.search(r'\btransaction year\s*(20\d{2})\b', q)
    if m:
        yr = int(m.group(1))
        filters.append({"column":"TransactionYear","operator":"=","value": yr})
        return filters
    m = re.search(r'\btransaction month\s*(\d{1,2})\b', q)
    if m:
        mon = int(m.group(1))
        filters.append({"column":"TransactionMonth","operator":"=","value": mon})
        return filters

    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        yr = int(m.group(1))
        filters.append({"column":"FinancialYear","operator":"=","value": yr})
        return filters

    return filters

# -------------------------
# Plan normalization (same behavior as your previous but robust)
# -------------------------
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)
def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None

def normalize_plan(plan: dict):
    plan = dict(plan or {})
    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []

    seen_aliases = set()
    clean_selects = []

    gb = []
    schema = get_schema()
    for g in group_by:
        if g and g not in gb and (not schema or g in schema):
            gb.append(g)

    gb_in_select_aliases = set()
    for s in selects:
        a = s.get("alias") or s.get("column")
        if a:
            gb_in_select_aliases.add(a)

    for g in gb:
        if g not in gb_in_select_aliases:
            clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})

    for s in selects:
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or (col if col else "value")
        alias = alias.replace("%", "_pct")
        if isinstance(expr, str) and is_aggregate_expression(expr):
            agg = None
        if agg is not None and (str(agg).lower() in ("none","null")):
            agg = None
        if alias in seen_aliases or (col and col in seen_aliases):
            continue
        seen_aliases.add(alias)
        if col:
            seen_aliases.add(col)
        clean_selects.append({"column": col, "expression": expr, "aggregation": agg, "alias": alias})

    final_selects = []
    seen = set()
    for s in clean_selects:
        key = s.get("alias") or s.get("column")
        if key and key not in seen:
            seen.add(key)
            final_selects.append(s)

    valid_order_by = []
    select_aliases = {s.get("alias") for s in final_selects if s.get("alias")}
    for ob in order_by:
        col = ob.get("column")
        if col in select_aliases or col in gb:
            valid_order_by.append(ob)

    plan["select"] = final_selects
    plan["group_by"] = gb
    plan["order_by"] = valid_order_by
    plan["filters"] = plan.get("filters", []) or []
    plan["limit"] = plan.get("limit")
    return plan

# -------------------------
# Fallback metric detector (if LLM returns incomplete plan)
# -------------------------
def detect_metric_from_text(text: str):
    text_low = (text or "").lower()
    # look for exact synonyms first
    for syn, metric in SYNONYM_MAP.items():
        if re.search(r'\b' + re.escape(syn) + r'\b', text_low):
            return metric
    # try simple substring match (looser)
    for syn, metric in SYNONYM_MAP.items():
        if syn in text_low:
            return metric
    return None

def build_plan_from_metric(metric_key, question_text):
    """
    Build a simple plan when LLM fails:
    - select metric expression with metric's aggregation (or expression)
    - add time filters (from question)
    """
    m = METRICS.get(metric_key)
    if not m:
        return None
    expr = m.get("expression")
    agg = m.get("aggregation")
    alias = metric_key
    select_item = {}
    # If expression is already an aggregate-style CASE or COUNT(...), keep as expression
    if isinstance(expr, str) and re.search(r'^\s*(count|sum|avg|min|max)\s*\(', expr.strip(), re.IGNORECASE):
        select_item = {"column": None, "expression": expr, "aggregation": None, "alias": alias}
    else:
        # if aggregation present, we will wrap expression inside aggregate
        if agg:
            select_item = {"column": None, "expression": expr, "aggregation": agg, "alias": alias}
        else:
            select_item = {"column": None, "expression": expr, "aggregation": None, "alias": alias}
    plan = {"select": [select_item], "filters": [], "group_by": [], "order_by": []}
    # attach any time filters parsed from question
    tf = parse_time_filters(question_text)
    for t in tf:
        t["value"] = sanitize_filter_value(t.get("value"))
    plan["filters"].extend(tf)
    return plan

# -------------------------
# Prompt builder
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- Financial Year rule: FY starts in March and ends in February next year.\n"
        "- When user mentions 'FY' or a plain 4-digit year, interpret as FinancialYear by default.\n"
        "- Use FinancialMonth/FinancialQuarter/FinancialYear columns when possible.\n"
        "- If user explicitly says 'transaction year/month', use TransactionYear/TransactionMonth.\n"
        "- Output STRICT valid JSON only. Do NOT include any explanation or extra text.\n"
        "- JSON keys must be: select (list), filters (list), group_by (list), order_by (list), limit (nullable).\n"
        "- Each select item: {\"column\": <column-name|null>, \"expression\": <sql-expr|null>, \"aggregation\": <SUM|AVG|COUNT|MIN|MAX|null>, \"alias\": <alias>}.\n"
    )

    return f"""
You are a SQL semantic engine. Convert the user's question into STRICT JSON query plan (no explanation).

{rules}

USER QUESTION:
{question}

TABLE: {TABLE_NAME}
SCHEMA: {json.dumps(schema or {}, indent=2)}
BUSINESS METRICS: {json.dumps(metrics, indent=2)}

OUTPUT ONLY valid JSON.
"""

# -------------------------
# Robust LLM call + JSON extraction
# -------------------------
def call_model_and_get_plan(client, model_name, prompt):
    """
    Calls the Groq client and tries to parse the JSON plan robustly.
    Returns: dict plan or raises Exception
    """
    if client is None:
        raise RuntimeError("Missing GROQ_API_KEY or unable to create Groq client")

    # Some models may occasionally inject extra text; we attempt multiple cleaning strategies
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
    except Exception as e:
        raise RuntimeError(f"GROQ model call failed: {e}")

    try:
        raw = resp.choices[0].message.content.strip()
    except Exception:
        raw = str(resp)

    # If the model printed debug / thought tokens, remove them: attempt to locate JSON
    # Approach: find first "{" and last "}" and take substring. Try to fix common issues.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        # no JSON detected
        raise ValueError(f"AI returned no JSON: {raw[:300]}")

    json_text = raw[start:end+1].strip()

    # Attempt 1: direct parse
    try:
        plan = json.loads(json_text)
        if isinstance(plan, dict):
            return plan
    except Exception:
        pass

    # Attempt 2: handle single quotes -> double quotes
    try:
        txt2 = re.sub(r"'", '"', json_text)
        plan = json.loads(txt2)
        if isinstance(plan, dict):
            return plan
    except Exception:
        pass

    # Attempt 3: remove trailing commas (common LLM bug)
    try:
        txt3 = re.sub(r",\s*}", "}", json_text)
        txt3 = re.sub(r",\s*]", "]", txt3)
        plan = json.loads(txt3)
        if isinstance(plan, dict):
            return plan
    except Exception:
        pass

    # Attempt 4: as a last resort, try to eval-like parse (very risky) — but we won't eval raw text.
    raise ValueError(f"AI returned JSON but parsing failed. Raw start: {raw[:400]}")

# -------------------------
# Main extraction function
# -------------------------
def extract_query(question: str):
    """
    Returns a normalized plan dict:
    { select: [...], filters: [...], group_by: [...], order_by: [...], limit: <int|null> }
    On error, returns a plan built from metric-detection fallback or includes an 'error' key.
    """
    client = get_client()
    schema = get_schema() or {}

    # Build prompt and call model
    prompt = build_prompt(question, schema, METADATA, METRICS)
    model_name = choose_best_groq_model(client) if client else None
    if not model_name and client:
        model_name = "qwen/qwen3-32b"

    # Try to call model and parse JSON plan
    plan = None
    raw_plan = None
    if client and model_name:
        try:
            raw_plan = call_model_and_get_plan(client, model_name, prompt)
            if isinstance(raw_plan, dict):
                plan = raw_plan
        except Exception as e:
            # model call / parse failure - log & fallback to metric detection
            print("⚠ Model parse error:", e)
            plan = None

    # If model failed, attempt a very small heuristic: see if metric is present in question
    if not plan:
        metric_key = detect_metric_from_text(question)
        if metric_key:
            print("ℹ Falling back to metric-detected plan for:", metric_key)
            plan = build_plan_from_metric(metric_key, question)
        else:
            # No metric detected — return an error-shaped plan to caller
            return {"error": "AI returned unparsable plan and no metric fallback available."}

    # Ensure plan is a dict
    if not isinstance(plan, dict):
        return {"error": "Plan returned is not a JSON object", "raw": plan}

    # Normalize filter values & append parsed time filters if they don't conflict
    time_filters = parse_time_filters(question)
    plan_filters = plan.get("filters", []) or []
    # append only if not duplicate columns
    for tf in time_filters:
        tf_value = sanitize_filter_value(tf.get("value"))
        tf["value"] = tf_value
        if not any(f.get("column") == tf.get("column") for f in plan_filters):
            plan_filters.append(tf)
    for f in plan_filters:
        f["value"] = sanitize_filter_value(f.get("value"))
    plan["filters"] = plan_filters

    # If selects is empty, but plan provided a 'metric' key or question contains metric, build fallback
    selects = plan.get("select", []) or []
    if not selects:
        metric_key = detect_metric_from_text(question)
        if metric_key:
            fb = build_plan_from_metric(metric_key, question)
            if fb:
                plan = fb

    # Normalize final plan
    plan = normalize_plan(plan)
    return plan
