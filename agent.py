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
# GROQ client & FIXED MODEL
# -------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None

# ✔ FIXED: ALWAYS USE qwen/qwen3-32b
def choose_best_groq_model(client):
    return "qwen/qwen3-32b"

# -------------------------
# Time utilities (FY starts in March)
# -------------------------
def calendar_to_fy_year(year:int, month:int) -> int:
    return year if month >= 3 else year - 1

def month_name_to_num(name: str):
    try:
        return datetime.strptime(name[:3].capitalize(), "%b").month
    except:
        try:
            return datetime.strptime(name, "%B").month
        except:
            return None

def calendar_to_fy_month(calendar_month:int) -> int:
    return ((calendar_month - 3) % 12) + 1

def fy_quarter_from_fy_month(fy_month:int) -> int:
    return ceil(fy_month / 3)

def current_utc():
    return datetime.utcnow()

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

# -------------------------
# Sanitize dict values
# -------------------------
def sanitize_filter_value(val):
    if isinstance(val, dict):
        for k in ("year", "quarter", "month", "value"):
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

    # FY 2024
    m = re.search(r'\b(?:fy|financial year)\s*(20\d{2})\b', q)
    if m:
        fy = int(m.group(1))
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    # Q1 2024
    m = re.search(r'\b(?:q|quarter)\s*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        qnum = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else None
        filters.append({"column":"FinancialQuarter","operator":"=","value": qnum})
        if year:
            filters.append({"column":"FinancialYear","operator":"=","value": year})
        return filters

    # Month + Year
    m = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
                  r'january|february|march|april|may|june|july|august|'
                  r'september|october|november|december)\s+(20\d{2})\b', 
                  q, flags=re.IGNORECASE)
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

    # last / previous month
    if "last month" in q or "previous month" in q:
        y, m = previous_calendar_month()
        fy = calendar_to_fy_year(y, m)
        fm = calendar_to_fy_month(m)
        filters.append({"column":"FinancialMonth","operator":"=","value": fm})
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    # last / previous quarter
    if "last quarter" in q or "previous quarter" in q:
        fy, fq = last_financial_quarter()
        filters.append({"column":"FinancialQuarter","operator":"=","value": fq})
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    # last / previous year
    if "last year" in q or "previous year" in q:
        prev = previous_calendar_year()
        filters.append({"column":"FinancialYear","operator":"=","value": prev})
        return filters

    # plain year
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

    schema = get_schema()
    gb = []
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
        if agg is not None and str(agg).lower() in ("none", "null"):
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
# Prompt builder
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- Financial Year rule: FY starts in March and ends in February next year.\n"
        "- Plain years → treat as FinancialYear unless user says transaction year.\n"
        "- Use FinancialMonth & FinancialQuarter when applicable.\n"
        "- Detect last/previous/compare/top N.\n"
    )

    return f"""
You are a SQL semantic engine. Convert the user's question into STRICT JSON query plan.

{rules}

USER QUESTION:
{question}

TABLE: {TABLE_NAME}
SCHEMA: {json.dumps(schema, indent=2)}
COLUMN DESCRIPTIONS: {json.dumps(metadata, indent=2)}
BUSINESS METRICS: {json.dumps(metrics, indent=2)}

OUTPUT ONLY JSON with:
select, filters, group_by, order_by, limit
"""

# -------------------------
# Extract query plan (LLM)
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
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"error": "LLM did not return JSON", "raw": raw}

    json_text = raw[start:end+1].replace("%", "_pct")

    try:
        plan = json.loads(json_text)
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}", "json_received": json_text, "raw": raw}

    time_filters = parse_time_filters(question)
    plan_filters = plan.get("filters", []) or []

    for tf in time_filters:
        tf["value"] = sanitize_filter_value(tf.get("value"))
        if not any(f.get("column") == tf.get("column") for f in plan_filters):
            plan_filters.append(tf)

    for f in plan_filters:
        f["value"] = sanitize_filter_value(f.get("value"))

    plan["filters"] = plan_filters

    plan = normalize_plan(plan)
    return plan
