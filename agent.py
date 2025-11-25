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

def choose_best_groq_model(client):
    preferred_order = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "llama-3.1-8b-instant"
    ]
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        for m in preferred_order:
            if m in available:
                return m
        for m in available:
            if "llama" in m or "qwen" in m:
                return m
        return available[0] if available else "llama-3.1-8b-instant"
    except Exception:
        return "llama-3.1-8b-instant"

# -------------------------
# Time utilities (FY starts in MARCH)
# Financial year mapping:
# FY X = Mar YEAR_X -> Feb YEAR_X+1
# -------------------------
def calendar_to_fy_year(year:int, month:int) -> int:
    # If month >= 3 (Mar..Dec), FY = year (i.e., Mar 2024 -> FY2024)
    # If month in Jan-Feb, FY = year -1 (Jan 2025 -> FY2024)
    return year if month >= 3 else year - 1

def month_name_to_num(name: str):
    try:
        # accept "Jan", "January", case-insensitive
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
    # returns list of (year, month) pairs for last n calendar months ending at now
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

# Given a calendar date get FY month index (1..12) where 1=Mar, 2=Apr,...12=Feb
def calendar_to_fy_month(calendar_month:int) -> int:
    return ((calendar_month - 3) % 12) + 1

def fy_quarter_from_fy_month(fy_month:int) -> int:
    return ceil(fy_month / 3)

# Convert a "last quarter" phrase -> (FinancialYear, FinancialQuarter)
def last_financial_quarter(reference=None):
    if not reference:
        reference = current_utc()
    fy_year = calendar_to_fy_year(reference.year, reference.month)
    fy_month = calendar_to_fy_month(reference.month)
    fq = fy_quarter_from_fy_month(fy_month)
    # previous quarter
    if fq == 1:
        return fy_year - 1, 4
    else:
        return fy_year, fq - 1

# previous calendar month
def previous_calendar_month(reference=None):
    if not reference:
        reference = current_utc()
    first = reference.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.year, prev.month

# previous calendar year
def previous_calendar_year(reference=None):
    if not reference:
        reference = current_utc()
    return reference.year - 1

# -------------------------
# Helper: sanitize filter value to be primitive (int/str)
# -------------------------
def sanitize_filter_value(val):
    # If dict with common keys, extract sensible primitive
    if isinstance(val, dict):
        # common possible forms: {"year":2024} or {"quarter":3} or {"month":5}
        for k in ("year","quarter","month","value"):
            if k in val:
                return val[k]
        # fallback: try to find an int in dict
        for v in val.values():
            if isinstance(v, int):
                return v
        # last resort: string convert
        return str(val)
    # If list, not directly supported—caller must handle
    if isinstance(val, list):
        return val
    return val

# -------------------------
# Time phrase parser
# -------------------------
def parse_time_filters(text: str):
    """
    Returns a list of filters like:
    { "column": "FinancialYear", "operator": "=", "value": 2024 }
    Uses FY by default for plain year (user specified FY default)
    Handles:
      - "last month", "previous month"
      - "last quarter", "previous quarter"
      - "last year", "previous year"
      - explicit month names + year (e.g., January 2024)
      - explicit quarters "Q1 2024" or "Quarter 1 2024"
      - "FY 2024" or "financial year 2024"
    """
    q = (text or "").lower()
    filters = []

    # explicit FY mention: "FY 2024" or "financial year 2024"
    m = re.search(r'\b(?:fy|financial year)\s*[:#-]?\s*(20\d{2})\b', q)
    if m:
        fy = int(m.group(1))
        # prefer FYYear column name if present (you mentioned FYYear exists)
        col = "FYYear" if "FYYear" in METADATA else ("FinancialYear" if "FinancialYear" in METADATA else "FinancialYear")
        filters.append({"column": col, "operator":"=", "value": fy})
        return filters

    # explicit quarter with optional year: "Q1 2024" or "Quarter 2 2023"
    m = re.search(r'\b(?:q|quarter)\s*[-:\s]*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        qnum = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else None
        if year:
            # since default is FY interpretation, map the year directly to FinancialYear
            filters.append({"column":"FinancialQuarter","operator":"=","value": qnum})
            filters.append({"column":"FinancialYear","operator":"=","value": year})
        else:
            filters.append({"column":"FinancialQuarter","operator":"=","value": qnum})
        return filters

    # explicit month + year: e.g., January 2024 or Jan 2024
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

    # last month / previous month
    if re.search(r'\blast month\b|\bprevious month\b', q):
        y, m = previous_calendar_month()
        fy = calendar_to_fy_year(y, m)
        fm = calendar_to_fy_month(m)
        filters.append({"column":"FinancialMonth","operator":"=","value": fm})
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    # last quarter / previous quarter (use FY logic)
    if re.search(r'\blast quarter\b|\bprevious quarter\b', q):
        fy, fq = last_financial_quarter()
        filters.append({"column":"FinancialQuarter","operator":"=","value": fq})
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    # last year / previous year
    if re.search(r'\blast year\b|\bprevious year\b', q):
        # interpret as FinancialYear by default
        prev = previous_calendar_year()
        filters.append({"column":"FinancialYear","operator":"=","value": prev})
        return filters

    # explicit calendar / transaction mention (if user says transaction year/month)
    # Example: "transaction year 2024" -> use TransactionYear
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

    # plain 4-digit year (no 'FY' word) -> per your preference treat as FinancialYear
    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        yr = int(m.group(1))
        filters.append({"column":"FinancialYear","operator":"=","value": yr})
        return filters

    # Nothing found
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
    """
    Normalize LLM plan:
     - Sanitize aliases (% -> _pct)
     - Ensure group_by is a list of unique columns
     - Ensure select contains group_by columns only once
     - If a select.expression already contains an aggregate, set aggregation -> None
     - Avoid nested aggregations: don't wrap expression if already aggregate
     - Ensure only one instance of each dimension in SELECT
    """
    plan = dict(plan)  # shallow copy
    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []

    # sanitize aliases and expressions
    seen_aliases = set()
    clean_selects = []

    # first, ensure group_by unique & valid
    gb = []
    schema = get_schema()
    for g in group_by:
        if g and g not in gb and g in schema:
            gb.append(g)

    # ensure select includes group_by columns (if not present in LLM, add them)
    gb_in_select_aliases = set()
    for s in selects:
        a = s.get("alias") or s.get("column")
        if a:
            gb_in_select_aliases.add(a)

    for g in gb:
        if g not in gb_in_select_aliases:
            # insert at beginning
            clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})

    for s in selects:
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or (col if col else "value")
        # sanitize alias % char
        alias = alias.replace("%", "_pct")
        # if expression contains aggregate, don't aggregate it again
        if isinstance(expr, str) and is_aggregate_expression(expr):
            agg = None
        # normalize aggregation value for JSON null/None
        if agg is not None and (str(agg).lower() == "none" or str(agg).lower() == "null"):
            agg = None
        # avoid duplicate dimension selects (if alias or column seen skip)
        if alias in seen_aliases or (col and col in seen_aliases):
            continue
        seen_aliases.add(alias)
        if col:
            seen_aliases.add(col)
        clean_selects.append({"column": col, "expression": expr, "aggregation": agg, "alias": alias})

    # dedupe clean_selects while preserving order
    final_selects = []
    seen = set()
    for s in clean_selects:
        key = s.get("alias") or s.get("column")
        if key and key not in seen:
            seen.add(key)
            final_selects.append(s)

    # normalize order_by - ensure column exists in final_selects or group_by
    valid_order_by = []
    select_aliases = {s.get("alias") for s in final_selects if s.get("alias")}
    for ob in order_by:
        col = ob.get("column")
        if col in select_aliases or col in gb:
            valid_order_by.append(ob)

    # final assembly
    plan["select"] = final_selects
    plan["group_by"] = gb
    plan["order_by"] = valid_order_by
    plan["filters"] = plan.get("filters", []) or []
    plan["limit"] = plan.get("limit")
    return plan

# -------------------------
# LLM prompt builder (includes FY rules)
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    # Add explicit FY rules & parsing instruction
    rules = (
        "- Financial Year rule: FY starts in March and ends in February next year.\n"
        "- When user mentions 'FY' or 'financial year' or when a plain year appears, interpret as FinancialYear by default.\n"
        "- FinancialMonth and FinancialQuarter columns exist; use them when producing time filters.\n"
        "- TransactionYear/TransactionMonth are calendar-based (Jan-Dec) and should be used only if user explicitly says 'transaction year' or 'transaction month'.\n"
        "- Detect 'last', 'previous', 'top N', 'compare', 'trend' and produce filters/groupings accordingly.\n"
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
        return {"error": "Missing GROQ_API_KEY"}

    schema = get_schema()
    prompt = build_prompt(question, schema, METADATA, METRICS)
    model_name = choose_best_groq_model(client)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    # new Groq API access
    raw = response.choices[0].message.content.strip()

    # get JSON block
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"error":"LLM did not return JSON", "raw": raw}
    json_text = raw[start:end+1]
    json_text = json_text.replace("%", "_pct")
    try:
        plan = json.loads(json_text)
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}", "json_received": json_text, "raw": raw}

    # Add time filters parsed from text (ensuring primitive values)
    time_filters = parse_time_filters(question)
    plan_filters = plan.get("filters", []) or []
    # append only if not duplicate columns and sanitize values
    for tf in time_filters:
        # sanitize tf value
        tf_value = sanitize_filter_value(tf.get("value"))
        tf["value"] = tf_value
        if not any(f.get("column")==tf.get("column") for f in plan_filters):
            plan_filters.append(tf)
    # sanitize any plan filters that may have dict-values
    for f in plan_filters:
        f["value"] = sanitize_filter_value(f.get("value"))
    plan["filters"] = plan_filters

    # Normalize plan (dedupe selects, remove nested agg, ensure group_by included)
    plan = normalize_plan(plan)
    return plan
