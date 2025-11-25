# agent.py — FINAL dynamic agent with FY & time parsing + plan normalization (updated, hardened)
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
# Helper: sanitize filter value to be primitive (int/str)
# -------------------------
def sanitize_filter_value(val):
    if isinstance(val, dict):
        for k in ("year","financialYear","quarter","month","value"):
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
# Parse simple filter string like "FinancialQuarter=3" or "TransactionYear = 2024"
# -------------------------
def parse_simple_filter_string(s: str):
    # accept "Column=Value", "Column = Value", "Column > 5"
    m = re.match(r'\s*([\w\[\]]+)\s*(=|>|<|>=|<=|!=)\s*(.+)\s*', s)
    if not m:
        return None
    col = m.group(1).strip()
    op = m.group(2)
    raw = m.group(3).strip().strip('"').strip("'")
    # try int
    try:
        val = int(raw)
    except:
        val = raw
    return {"column": col, "operator": op, "value": val}

# -------------------------
# Time phrase parser
# -------------------------
def parse_time_filters(text: str):
    q = (text or "").lower()
    filters = []

    m = re.search(r'\b(?:fy|financial year)\s*[:#-]?\s*(20\d{2})\b', q)
    if m:
        fy = int(m.group(1))
        filters.append({"column": "FinancialYear", "operator":"=", "value": fy})
        return filters

    if re.search(r'\blast quarter\b|\bprevious quarter\b', q):
        fy, fq = last_financial_quarter()
        filters.append({"column":"FinancialQuarter","operator":"=","value": fq})
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    if re.search(r'\blast month\b|\bprevious month\b', q):
        y, m = previous_calendar_month()
        fy = calendar_to_fy_year(y, m)
        fm = calendar_to_fy_month(m)
        filters.append({"column":"FinancialMonth","operator":"=","value": fm})
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    if re.search(r'\blast year\b|\bprevious year\b', q):
        prev = previous_calendar_year()
        filters.append({"column":"FinancialYear","operator":"=","value": prev})
        return filters

    m = re.search(r'\b(?:q|quarter)\s*[-:\s]*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        qnum = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else None
        filters.append({"column":"FinancialQuarter","operator":"=","value": qnum})
        if year:
            filters.append({"column":"FinancialYear","operator":"=","value": year})
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
# Plan cleaning / normalization helpers
# -------------------------
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)
def is_aggregate_expression(expr: str) -> bool:
    return isinstance(expr, str) and AGG_RE.match(expr.strip()) is not None

DIMENSION_MAP = {
    "department": "DeptCode",
    "dept": "DeptCode",
    "branch": "BranchCode",
    "company": "CompanyCode",
    "customer": "CustomerLeadGroup",
    "customer group": "CustomerLeadGroup",
    "customerleadgroup": "CustomerLeadGroup"
}

PK_COLUMNS = {
    "JobPK", "CountryPK", "CompanyPK",
    "BranchPK", "DeptPK"
}

def coerce_select_item(item):
    # item could be dict, or string representing column, or malformed JSON
    if isinstance(item, dict):
        col = item.get("column") or item.get("col") or item.get("c")
        expr = item.get("expression")
        agg = item.get("aggregation")
        alias = item.get("alias") or col or (expr and expr[:20]) or "value"
        return {"column": col, "expression": expr, "aggregation": agg, "alias": alias}
    if isinstance(item, str):
        # if string like "SUM([JobProfit]) AS profit" try to parse alias, else treat as column name
        m = re.match(r'(.+)\s+as\s+([\w\[\]]+)$', item, re.IGNORECASE)
        if m:
            expr = m.group(1).strip()
            alias = m.group(2).strip()
            # try to extract column from simple "[Col]" or bare word
            col_m = re.search(r'\[?([A-Za-z0-9_]+)\]?', expr)
            col = col_m.group(1) if col_m else None
            return {"column": col, "expression": expr, "aggregation": None, "alias": alias}
        # bare column
        return {"column": item, "expression": None, "aggregation": None, "alias": item}
    # unknown -> return safe default
    return {"column": None, "expression": None, "aggregation": None, "alias": "value"}

def coerce_filter_item(item):
    # item could be dict already, or string "FinancialQuarter=3", or malformed
    if isinstance(item, dict):
        # ensure keys exist
        col = item.get("column") or item.get("col") or item.get("c")
        op = item.get("operator") or item.get("op") or "="
        val = sanitize_filter_value(item.get("value"))
        return {"column": col, "operator": op, "value": val}
    if isinstance(item, str):
        parsed = parse_simple_filter_string(item)
        if parsed:
            parsed["value"] = sanitize_filter_value(parsed["value"])
            return parsed
        # fallback: entire string as value (not ideal), skip
        return None
    return None

def coerce_order_item(item):
    if isinstance(item, dict):
        col = item.get("column")
        dirn = item.get("direction") or item.get("dir") or "DESC"
        return {"column": col, "direction": dirn.upper()}
    if isinstance(item, str):
        # "Profit DESC" or "Profit"
        m = re.match(r'([\w\[\]]+)(?:\s+(asc|desc))?', item, re.IGNORECASE)
        if m:
            return {"column": m.group(1), "direction": (m.group(2) or "DESC").upper()}
    return None

def normalize_plan(plan: dict):
    # ensure plan is dict
    if not isinstance(plan, dict):
        return {"select": [], "filters": [], "group_by": [], "order_by": [], "limit": None}

    schema = get_schema()

    # coerce primary keys
    selects_raw = plan.get("select", []) or []
    filters_raw = plan.get("filters", []) or []
    group_by_raw = plan.get("group_by", []) or []
    order_by_raw = plan.get("order_by", []) or []

    # coerce lists
    if isinstance(selects_raw, dict):
        # sometimes LLM outputs a single object
        selects_raw = [selects_raw]
    if isinstance(filters_raw, dict):
        filters_raw = [filters_raw]
    if isinstance(group_by_raw, (str, dict)):
        group_by_raw = [group_by_raw]
    if isinstance(order_by_raw, (str, dict)):
        order_by_raw = [order_by_raw]

    # coerce group_by to list of column strings
    group_by = []
    for g in group_by_raw:
        if isinstance(g, str):
            gcol = g
        elif isinstance(g, dict):
            gcol = g.get("column") or g.get("col") or g.get("alias")
        else:
            gcol = None
        if gcol:
            # dimension mapping
            if gcol.lower() in DIMENSION_MAP:
                gcol = DIMENSION_MAP[gcol.lower()]
            if gcol not in PK_COLUMNS and gcol in schema and gcol not in group_by:
                group_by.append(gcol)

    # coerce selects
    clean_selects = []
    seen = set()
    # ensure group_by columns come first in selects
    for g in group_by:
        clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})
        seen.add(g)

    for s in selects_raw:
        coerced = coerce_select_item(s)
        col = coerced.get("column")
        alias = coerced.get("alias") or col or "value"
        # dimension mapping for alias/col
        if isinstance(alias, str) and alias.lower() in DIMENSION_MAP:
            alias = DIMENSION_MAP[alias.lower()]
            coerced["alias"] = alias
            coerced["column"] = alias
            col = alias

        # skip PK columns
        if col in PK_COLUMNS:
            continue

        # avoid duplicate columns/aliases
        if alias in seen or (col and col in seen):
            continue
        seen.add(alias)
        if col:
            seen.add(col)

        # avoid wrapping aggregated expressions
        if coerced.get("expression") and is_aggregate_expression(coerced.get("expression")):
            coerced["aggregation"] = None

        # normalize aggregation null strings
        agg = coerced.get("aggregation")
        if agg is not None and str(agg).lower() in ("none","null"):
            coerced["aggregation"] = None

        clean_selects.append(coerced)

    # coerce filters
    clean_filters = []
    for f in filters_raw:
        coerced = coerce_filter_item(f)
        if not coerced:
            continue
        col = coerced.get("column")
        # map dimension names (customer, branch, department, company) if user used words
        if isinstance(col, str) and col.lower() in DIMENSION_MAP:
            coerced["column"] = DIMENSION_MAP[col.lower()]
            col = coerced["column"]
        # ensure not PK column and exists in schema
        if col and col in schema and col not in PK_COLUMNS:
            coerced["value"] = sanitize_filter_value(coerced.get("value"))
            clean_filters.append(coerced)

    # coerce order_by
    clean_order = []
    for ob in order_by_raw:
        coerced = coerce_order_item(ob)
        if coerced:
            clean_order.append(coerced)

    # limit coercion
    limit = plan.get("limit")
    try:
        if isinstance(limit, str):
            limit = int(limit)
    except:
        limit = None

    normalized = {
        "select": clean_selects,
        "filters": clean_filters,
        "group_by": group_by,
        "order_by": clean_order,
        "limit": limit
    }
    return normalized

# -------------------------
# LLM prompt builder (includes FY rules)
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- Financial Year rule: FY starts in March and ends in February next year.\n"
        "- When user mentions 'FY' or 'financial year' or when a plain year appears, interpret as FinancialYear by default.\n"
        "- FinancialMonth and FinancialQuarter columns exist; use them when producing time filters.\n"
        "- TransactionYear/TransactionMonth are calendar-based (Jan-Dec) and should be used only if user explicitly says 'transaction year' or 'transaction month'.\n"
        "- Detect 'last', 'previous', 'top N', 'compare', 'trend' and produce filters/groupings accordingly.\n"
        "- NEVER output objects like {\"function\":..., \"args\":...} in filter values. Use literal numbers.\n"
        "- Map natural dimensions: department->DeptCode, branch->BranchCode, company->CompanyCode, customer->CustomerLeadGroup.\n"
        "- Do not reference PK columns (JobPK, DeptPK, BranchPK, CompanyPK, CountryPK).\n"
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

    # New Groq API access
    raw = ""
    try:
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"error": f"LLM response access error: {e}"}

    # find JSON block
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"error":"LLM did not return JSON", "raw": raw}
    json_text = raw[start:end+1]

    # LLM sometimes double-encodes JSON as a string; attempt to parse robustly
    plan = None
    try:
        plan = json.loads(json_text)
    except Exception:
        # try removing escape quotes if it's a JSON string
        try:
            unq = json_text.strip().strip('"').replace('\\"', '"').replace("\\'", "'")
            plan = json.loads(unq)
        except Exception:
            # fallback: return parse error with raw for debugging
            return {"error": "Failed to parse LLM JSON", "json_text": json_text, "raw": raw}

    # plan may still be a string or list - normalize it into expected shape
    # Merge time filters parsed from text, but ensure values are primitives
    time_filters = parse_time_filters(question)
    # coerce time_filters values
    for tf in time_filters:
        tf["value"] = sanitize_filter_value(tf.get("value"))

    # If plan contains filters as strings, coerce later in normalize_plan
    # Merge plan filters and time filters ensuring no column duplicates
    existing_filters = plan.get("filters", []) if isinstance(plan, dict) else []
    if isinstance(existing_filters, dict):
        existing_filters = [existing_filters]
    merged_filters = []
    # sanitize and append existing
    for f in existing_filters:
        coerced = coerce_filter_item(f) if not isinstance(f, str) else coerce_filter_item(f)
        if coerced:
            merged_filters.append(coerced)
    # append time_filters if column not present
    for tf in time_filters:
        if not any(f.get("column") == tf.get("column") for f in merged_filters):
            merged_filters.append(tf)

    # put merged filters back into plan (coerce other fields similarly)
    tentative_plan = {}
    tentative_plan["select"] = plan.get("select", []) if isinstance(plan, dict) else []
    tentative_plan["filters"] = merged_filters
    tentative_plan["group_by"] = plan.get("group_by", []) if isinstance(plan, dict) else []
    tentative_plan["order_by"] = plan.get("order_by", []) if isinstance(plan, dict) else []
    tentative_plan["limit"] = plan.get("limit") if isinstance(plan, dict) else None

    # Finally normalize into strict types and validate
    normalized = normalize_plan(tentative_plan)

    return normalized
