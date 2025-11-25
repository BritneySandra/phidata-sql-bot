# agent.py — FINAL dynamic agent with FY & time parsing + plan normalization (updated)
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
# GROQ client & auto-model
# -------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None

# ✅ UPDATED MODEL BLOCK (new)
def choose_best_groq_model(client):
    # Updated preferred order: prioritize qwen/qwen3-32b
    preferred_order = [
        "qwen/qwen3-32b",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ]
    try:
        models = client.models.list()
        available = [m.id for m in models.data]

        # Choose best from preferred list
        for m in preferred_order:
            if m in available:
                return m

        # Fallback: any llama or qwen
        for m in available:
            if "llama" in m or "qwen" in m:
                return m

        # Final fallback
        return available[0] if available else "qwen/qwen3-32b"
    except Exception:
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

def calendar_month_from_text(token: str):
    token = token.strip().lower()
    short = token[:3].capitalize()
    try:
        dt = datetime.strptime(short, "%b")
        return dt.month
    except:
        return None

def current_utc():
    return datetime.utcnow()

def last_n_months_period(n:int):
    now = current_utc()
    months = []
    y = now.year
    m = now.month
    for _ in range(n):
        months.append((y,m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return months[::-1]

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
# sanitize filter values
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
# Time phrase parsing
# -------------------------
def parse_time_filters(text: str):
    q = (text or "").lower()
    filters = []

    m = re.search(r'\b(?:fy|financial year)\s*[:#-]?\s*(20\d{2})\b', q)
    if m:
        fy = int(m.group(1))
        col = "FYYear" if "FYYear" in METADATA else "FinancialYear"
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

    m = re.search(
        r'\b('
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
# Plan normalization
# -------------------------
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)
def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None

def normalize_plan(plan: dict):
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

        if agg is not None and (str(agg).lower() == "none" or str(agg).lower() == "null"):
            agg = None

        if alias in seen_aliases or (col and col in seen_aliases):
            continue

        seen_aliases.add(alias)
        if col:
            seen_aliases.add(col)

        clean_selects.append({
            "column": col,
            "expression": expr,
            "aggregation": agg,
            "alias": alias
        })

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
# LLM prompt builder
# -------------------------
def build_prompt(question, schema, metadata, metrics):

    rules = (
        "- Financial Year rule: FY starts in March and ends in February next year.\n"
        "- Plain year → treat as FinancialYear.\n"
        "- Use FinancialMonth/FinancialQuarter when possible.\n"
        "- Use TransactionYear/TransactionMonth only if explicitly asked.\n"
        "- Detect: last, previous, top N, compare, trend.\n"
    )

    return f"""
You are a SQL semantic engine. Convert the user's question into STRICT JSON query plan only.

{rules}

USER QUESTION:
{question}

TABLE: {TABLE_NAME}
SCHEMA: {json.dumps(schema, indent=2)}
COLUMN DESCRIPTIONS: {json.dumps(metadata, indent=2)}
BUSINESS METRICS: {json.dumps(metrics, indent=2)}

OUTPUT ONLY valid JSON with keys:
select, filters, group_by, order_by, limit

Return JSON ONLY.
"""

# -------------------------
# Extract + normalize plan
# -------------------------
def extract_query(question: str):
    client = get_client()
    if not client:
        return {"error": "Missing GROQ_API_KEY"}

    schema = get_schema()
    prompt = build_prompt(question, schema, METADATA, METRICS)
    model_name = choose_best_groq_model(client)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"error":"LLM did not return JSON", "raw": raw}

    json_text = raw[start:end+1]
    json_text = json_text.replace("%", "_pct")

    try:
        plan = json.loads(json_text)
    except Exception as e:
        return {
            "error": f"Failed to parse JSON: {e}",
            "json_received": json_text,
            "raw": raw
        }

    time_filters = parse_time_filters(question)
    plan_filters = plan.get("filters", []) or []

    for tf in time_filters:
        tf["value"] = sanitize_filter_value(tf.get("value"))
        if not any(f.get("column")==tf.get("column") for f in plan_filters):
            plan_filters.append(tf)

    for f in plan_filters:
        f["value"] = sanitize_filter_value(f.get("value"))

    plan["filters"] = plan_filters

    plan = normalize_plan(plan)
    return plan
