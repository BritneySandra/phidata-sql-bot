# agent.py — dynamic agent with FY & time parsing + robust JSON extraction
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
    except Exception:
        return {}

METADATA = load_json_file("metadata.json")
METRICS = load_json_file("metrics.json")

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
# GROQ client & model selection (prefer qwen)
# -------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None

def choose_best_groq_model(client):
    # prefer qwen first as requested
    preferred_order = [
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ]
    try:
        models_resp = client.models.list()
        available = [m.id for m in models_resp.data]
        for p in preferred_order:
            if p in available:
                return p
        # fallback: pick a llama/qwen if available, else first
        for a in available:
            if "qwen" in a or "llama" in a:
                return a
        return available[0] if available else "qwen/qwen3-32b"
    except Exception as e:
        print("⚠ choose_best_groq_model failed:", e)
        # fallback to qwen (user requested)
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
# Helper: sanitize filter value to be primitive (int/str)
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
        col = "FYYear" if "FYYear" in METADATA else ("FinancialYear" if "FinancialYear" in METADATA else "FinancialYear")
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
        mon = m.group(1); yr = int(m.group(2))
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
# Plan normalizer (same logic but defensive)
# -------------------------
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)
def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None

def normalize_plan(plan: dict):
    if not isinstance(plan, dict):
        return {"error":"Plan is not a dict"}
    plan = dict(plan)
    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []

    seen_aliases = set()
    clean_selects = []

    gb = []
    schema = get_schema()
    for g in group_by:
        if g and g not in gb and g in schema:
            gb.append(g)

    gb_in_select_aliases = set()
    for s in selects:
        if isinstance(s, dict):
            a = s.get("alias") or s.get("column")
            if a:
                gb_in_select_aliases.add(a)

    for g in gb:
        if g not in gb_in_select_aliases:
            clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})

    for s in selects:
        if not isinstance(s, dict):
            continue
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
        if not isinstance(ob, dict):
            continue
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
# Robust JSON extraction from LLM raw text
# -------------------------
def extract_json_block(raw: str):
    if not raw or not isinstance(raw, str):
        return None
    # find first '{' and match braces to find balanced JSON
    start = raw.find("{")
    if start == -1:
        return None
    stack = []
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == "{":
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack:
                    return raw[start:i+1]
    return None

# -------------------------
# LLM prompt builder
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- Financial Year rule: FY starts in March and ends in February next year.\n"
        "- When user mentions 'FY' or 'financial year' or when a plain year appears, interpret as FinancialYear by default.\n"
        "- FinancialMonth and FinancialQuarter columns exist; use them when producing time filters.\n"
        "- TransactionYear/TransactionMonth are calendar-based (Jan-Dec) and should be used only if user explicitly says 'transaction year' or 'transaction month'.\n"
        "- Output ONLY valid JSON object with keys: select, filters, group_by, order_by, limit. Do NOT add any extra commentary.\n"
    )

    return f"""
You are a SQL semantic engine. Convert the user's question into STRICT JSON query plan (no explanation).

{rules}

USER QUESTION:
{question}

TABLE: {TABLE_NAME}
SCHEMA: {json.dumps(schema, indent=2)}
COLUMN DESCRIPTIONS: {json.dumps(metadata, indent=2)}
BUSINESS METRICS: {json.dumps(metrics, indent=2)}

OUTPUT ONLY valid JSON with keys:
select, filters, group_by, order_by, limit

select items: column (or null), expression (or null), aggregation (or null), alias

Time guidance:
- Use FinancialYear/FinancialMonth/FinancialQuarter when possible (FY starts March).
- Use TransactionYear/TransactionMonth only if user explicitly asks for transaction/calendar month/year.

Return JSON only.
"""

# -------------------------
# Extract and normalize plan
# -------------------------
def extract_query(question: str):
    client = get_client()
    if not client:
        return {"error":"Missing GROQ_API_KEY"}

    schema = get_schema()
    prompt = build_prompt(question, schema, METADATA, METRICS)
    model_name = choose_best_groq_model(client)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        raw = ""
        try:
            raw = response.choices[0].message.content.strip()
        except Exception:
            raw = str(response)
    except Exception as e:
        # model call failed
        print("⚠ LLM call failed:", e)
        return {"error": f"LLM call failed: {e}"}

    # extract JSON block robustly
    json_block = extract_json_block(raw)
    if not json_block:
        # give the raw response in debug to help trace issues
        return {"error":"ai_json_parse_failed", "raw": raw}

    # normalize '%' in aliases
    json_block = json_block.replace("%", "_pct")
    try:
        plan = json.loads(json_block)
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}", "json_received": json_block, "raw": raw}

    # Add time filters parsed from text
    time_filters = parse_time_filters(question)
    plan_filters = plan.get("filters", []) or []
    for tf in time_filters:
        tf_value = sanitize_filter_value(tf.get("value"))
        tf["value"] = tf_value
        if not any(f.get("column")==tf.get("column") for f in plan_filters):
            plan_filters.append(tf)
    for f in plan_filters:
        f["value"] = sanitize_filter_value(f.get("value"))
    plan["filters"] = plan_filters

    plan = normalize_plan(plan)
    return plan
