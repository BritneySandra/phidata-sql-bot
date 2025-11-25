# agent.py — robust dynamic agent (Option C: accept both aggregation formats)
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
# SQL Schema loader
# -------------------------
_SCHEMA = {}
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

def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA

# -------------------------
# Groq client & model selection
# -------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None

def choose_best_groq_model(client):
    preferred_order = [
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "allam-2-7b",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ]
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        for m in preferred_order:
            if m in available:
                return m
        for m in available:
            if ("qwen" in m.lower() or "llama" in m.lower() or "allam" in m.lower()) and "whisper" not in m.lower():
                return m
        return available[0] if available else preferred_order[0]
    except Exception:
        return preferred_order[0]

# -------------------------
# Time utilities (FY starts in March)
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

def calendar_to_fy_month(calendar_month:int) -> int:
    return ((calendar_month - 3) % 12) + 1

def fy_quarter_from_fy_month(fy_month:int) -> int:
    return ceil(fy_month / 3)

def current_utc():
    return datetime.utcnow()

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
# sanitize helpers
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
    if isinstance(val, (list, tuple)):
        return list(val)
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
# JSON extraction helper
# -------------------------
def extract_first_json_object(text: str):
    if not text or "{" not in text:
        return None
    start_positions = [m.start() for m in re.finditer(r'\{', text)]
    for start in start_positions:
        depth = 0
        i = start
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
            i += 1
    return None

# -------------------------
# Plan normalization (accept both aggregation styles)
# -------------------------
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)
def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None

def normalize_plan(plan: dict):
    # Ensure plan is object
    if not isinstance(plan, dict):
        return {"select": [], "filters": [], "group_by": [], "order_by": [], "limit": None}

    plan = dict(plan)
    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    filters = plan.get("filters", []) or []

    # Normalize types
    if not isinstance(selects, list):
        # If model returned a single select as dict or string -> wrap
        if isinstance(selects, dict):
            selects = [selects]
        else:
            selects = []

    if not isinstance(group_by, list):
        group_by = [group_by] if group_by else []

    if not isinstance(order_by, list):
        order_by = [order_by] if order_by else []

    if not isinstance(filters, list):
        filters = [filters] if filters else []

    schema = get_schema()
    seen = set()
    clean_selects = []

    # Ensure group_by columns are valid and unique
    clean_group_by = []
    for g in group_by:
        if isinstance(g, str) and g in schema and g not in clean_group_by:
            clean_group_by.append(g)

    # Prepend missing group_by columns to select (so UI shows them)
    for g in clean_group_by:
        clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})

    for s in selects:
        if not isinstance(s, dict):
            continue
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or (col if col else None)

        # Accept both styles: if expression contains aggregate, leave it; if 'aggregation' exists use it.
        if agg is not None and isinstance(agg, str) and agg.lower() in ("none", "null"):
            agg = None

        # if expression present and contains agg, don't double-wrap
        if isinstance(expr, str) and is_aggregate_expression(expr):
            agg = None

        if not alias:
            alias = col if col else (expr if isinstance(expr, str) else "value")

        # avoid duplicates
        key = alias or col
        if key in seen:
            continue
        seen.add(key)

        clean_selects.append({"column": col, "expression": expr, "aggregation": agg, "alias": alias})

    # Normalize filters: ensure list of dicts with primitive values
    clean_filters = []
    for f in filters:
        if isinstance(f, dict) and f.get("column"):
            f["value"] = sanitize_filter_value(f.get("value"))
            clean_filters.append(f)
        elif isinstance(f, str) and "=" in f:
            parts = f.split("=", 1)
            clean_filters.append({"column": parts[0].strip(), "operator": "=", "value": parts[1].strip().strip("'\"")})
        # ignore malformed

    # normalize order_by entries
    clean_order_by = []
    for ob in order_by:
        if isinstance(ob, dict):
            col = ob.get("column")
            dirn = ob.get("direction", "DESC") or "DESC"
            if col:
                clean_order_by.append({"column": col, "direction": dirn.upper()})
        elif isinstance(ob, str):
            # allow "col desc"
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*(asc|desc)?\s*$", ob, flags=re.IGNORECASE)
            if m:
                clean_order_by.append({"column": m.group(1), "direction": (m.group(2) or "DESC").upper()})

    normalized = {
        "select": clean_selects,
        "filters": clean_filters,
        "group_by": clean_group_by,
        "order_by": clean_order_by,
        "limit": plan.get("limit")
    }
    return normalized

# -------------------------
# Prompt builder (strict JSON)
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- Financial Year: FY starts in March and ends in February next year.\n"
        "- Plain year -> FinancialYear by default unless user mentions 'transaction'.\n"
        "- Use FinancialMonth/FinancialQuarter when possible; TransactionYear/TransactionMonth only when explicitly asked.\n"
        "- Detect last/previous/top N/compare/trend and produce filters/groupings accordingly.\n"
    )
    strict_json_instruction = (
        "YOU MUST OUTPUT ONE VALID JSON OBJECT ONLY (no explanation, no markdown, no extra text).\n"
        "If you cannot produce a plan, output exactly: {\"select\": [], \"filters\": [], \"group_by\": [], \"order_by\": [], \"limit\": null}\n"
        "Types: select:list, filters:list, group_by:list, order_by:list, limit:null|int.\n"
    )
    return f"""
You are a SQL semantic engine. Convert the user's question into STRICT JSON query plan only.

{rules}
{strict_json_instruction}

USER QUESTION:
{question}

TABLE: {TABLE_NAME}
SCHEMA: {json.dumps(schema, indent=2)}
COLUMN DESCRIPTIONS: {json.dumps(metadata, indent=2)}
BUSINESS METRICS: {json.dumps(metrics, indent=2)}

OUTPUT JSON keys:
select, filters, group_by, order_by, limit

select items objects:
- column (string|null)
- expression (string|null)   # SQL expression, may already include aggregate, e.g. "SUM(JobProfit)"
- aggregation (string|null)  # e.g., "sum","avg" - optional
- alias (string)             # output column alias

filters objects:
{ "column": "<col>", "operator": "=", "value": <primitive> }

Return JSON only.
"""

# -------------------------
# Main: extract_query(question) -> normalized plan
# -------------------------
def extract_query(question: str):
    client = get_client()
    if not client:
        return {"error": "Missing GROQ_API_KEY"}

    schema = get_schema()
    prompt = build_prompt(question, schema, METADATA, METRICS)
    model_name = choose_best_groq_model(client)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
    except Exception as e:
        return {"error": f"LLM request failed: {e}"}

    # read raw safely
    raw = ""
    try:
        raw = response.choices[0].message.content.strip()
    except Exception:
        try:
            raw = str(response.choices[0].text).strip()
        except Exception:
            raw = str(response)

    # extract first valid JSON object
    json_text = extract_first_json_object(raw)
    if not json_text:
        # fallback naive slice
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end+1]
            try:
                json.loads(candidate)
                json_text = candidate
            except Exception:
                return {"error": "LLM did not return valid JSON", "raw": raw}

    try:
        plan = json.loads(json_text)
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}", "json_received": json_text, "raw": raw}

    if not isinstance(plan, dict):
        return {"error": "LLM returned JSON but not an object", "raw": plan}

    # Merge time filters parsed from text
    time_filters = parse_time_filters(question)
    plan_filters = plan.get("filters", []) or []
    if not isinstance(plan_filters, list):
        plan_filters = [plan_filters] if plan_filters else []

    # sanitize existing filters then append time filters if missing
    sanitized = []
    for f in plan_filters:
        if isinstance(f, dict) and f.get("column"):
            f["value"] = sanitize_filter_value(f.get("value"))
            sanitized.append(f)
        elif isinstance(f, str) and "=" in f:
            parts = f.split("=", 1)
            sanitized.append({"column": parts[0].strip(), "operator": "=", "value": parts[1].strip().strip("'\"")})

    for tf in time_filters:
        tf_val = sanitize_filter_value(tf.get("value"))
        tf["value"] = tf_val
        if not any(isinstance(f, dict) and f.get("column") == tf.get("column") for f in sanitized):
            sanitized.append(tf)

    plan["filters"] = sanitized

    # Normalize plan (dedupe selects, accept both agg styles)
    plan = normalize_plan(plan)
    return plan
