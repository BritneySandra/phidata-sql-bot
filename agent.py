# agent.py — FINAL dynamic agent with FY & time parsing + plan normalization (updated & hardened)
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
    # Prefer qwen first (stable + efficient), then other reasonable fallbacks.
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
        # fallback: pick any LLM-like model
        for m in available:
            if ("qwen" in m.lower() or "llama" in m.lower() or "allam" in m.lower()) and "guard" not in m.lower() and "whisper" not in m.lower() and "tts" not in m.lower():
                return m
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
    # FY months: 1=Mar, 2=Apr, ..., 12=Feb
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
        # leave lists intact (e.g., IN clauses) — but caller must support lists
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
    # defensive: ensure plan is a dict
    if not isinstance(plan, dict):
        return {"select": [], "filters": [], "group_by": [], "order_by": [], "limit": None}

    plan = dict(plan)  # shallow copy
    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []

    # ensure types
    if not isinstance(selects, list):
        selects = []
    if not isinstance(group_by, list):
        group_by = []
    if not isinstance(order_by, list):
        order_by = []

    # sanitize aliases and expressions
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
            # ignore malformed select entry
            continue
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or (col if col else "value")

        alias = alias.replace("%", "_pct") if isinstance(alias, str) else alias

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

    # ensure filters is a list of dicts
    clean_filters = []
    for f in plan["filters"]:
        if isinstance(f, dict):
            # sanitize value
            f["value"] = sanitize_filter_value(f.get("value"))
            clean_filters.append(f)
        else:
            # if filter is string like "Customer=ABC" try parse rudimentarily
            if isinstance(f, str) and "=" in f:
                parts = f.split("=", 1)
                col = parts[0].strip()
                val = parts[1].strip()
                clean_filters.append({"column": col, "operator": "=", "value": val})
            # else ignore malformed filter entries
    plan["filters"] = clean_filters

    return plan

# -------------------------
# Improved JSON extraction helper
# -------------------------
def extract_first_json_object(text: str):
    """
    Scans text and returns the first balanced JSON object string it finds.
    Uses a stack-based brace counting approach to handle nested braces.
    Returns None if not found.
    """
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
                    # try parse
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        # candidate not valid JSON (maybe trailing commas etc) — continue scanning
                        break
            i += 1
    return None

# -------------------------
# LLM prompt builder (includes FY & strict JSON instruction)
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- Financial Year rule: FY starts in March and ends in February next year.\n"
        "- Treat plain year as FinancialYear by default unless user explicitly states 'transaction'.\n"
        "- Use FinancialMonth/FinancialQuarter when possible.\n"
        "- TransactionYear/TransactionMonth are calendar-based (Jan-Dec) and should be used only if explicitly asked.\n"
        "- Detect: last/previous/top N/compare/trend and produce filters/groupings accordingly.\n"
    )

    # Strong instruction to produce JSON only and exact fallback
    strict_json_instruction = (
        "YOU MUST OUTPUT ONE VALID JSON OBJECT ONLY, and NOTHING ELSE (no explanation, no markdown, no extra text).\n"
        "If you cannot produce a plan, output exactly: {\"select\": [], \"filters\": [], \"group_by\": [], \"order_by\": [], \"limit\": null}\n"
        "Ensure types: select:list, filters:list, group_by:list, order_by:list, limit:null|int.\n"
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

OUTPUT ONLY valid JSON with keys:
select, filters, group_by, order_by, limit

select items should be objects with keys:
- column (string or null)
- expression (string or null)  # optional SQL expression using column names
- aggregation (string or null) # e.g., sum, avg, count
- alias (string)

filters are objects: { "column": "<col>", "operator": "=", "value": <primitive> }

Return JSON ONLY.
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

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
    except Exception as e:
        return {"error": f"LLM request failed: {e}"}

    # Support both shapes: some SDKs return choices with .message.content, others slightly different
    raw = ""
    try:
        raw = response.choices[0].message.content.strip()
    except Exception:
        try:
            # fallback for different response shape
            raw = str(response.choices[0].text).strip()
        except Exception:
            raw = str(response)

    # Try to extract the first valid JSON object
    json_text = extract_first_json_object(raw)
    if not json_text:
        # last resort: naive brace slice (previous approach), but we still must validate
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end+1]
            try:
                json.loads(candidate)
                json_text = candidate
            except Exception:
                # give up and return raw for debugging
                return {"error": "LLM did not return valid JSON", "raw": raw}

    try:
        plan = json.loads(json_text)
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}", "json_received": json_text, "raw": raw}

    # Defensive sanitization: ensure plan is a dict
    if not isinstance(plan, dict):
        return {"error": "LLM returned JSON but not an object", "raw_json": plan}

    # Add time filters parsed from text (ensuring primitive values)
    time_filters = parse_time_filters(question)
    plan_filters = plan.get("filters", []) or []

    if not isinstance(plan_filters, list):
        # if LLM produced filters as a string or dict, normalize
        if isinstance(plan_filters, dict):
            plan_filters = [plan_filters]
        else:
            plan_filters = []

    # append only if not duplicate columns and sanitize values
    for tf in time_filters:
        tf_value = sanitize_filter_value(tf.get("value"))
        tf["value"] = tf_value
        if not any(isinstance(f, dict) and f.get("column") == tf.get("column") for f in plan_filters):
            plan_filters.append(tf)

    # sanitize any plan filters that may have dict-values or malformed entries
    sanitized_filters = []
    for f in plan_filters:
        if isinstance(f, dict):
            f["value"] = sanitize_filter_value(f.get("value"))
            if "column" in f:
                sanitized_filters.append(f)
        else:
            # try a basic string parse like "TransportMode = SEA"
            if isinstance(f, str) and "=" in f:
                parts = f.split("=", 1)
                col = parts[0].strip()
                val = parts[1].strip().strip("'\"")
                sanitized_filters.append({"column": col, "operator": "=", "value": val})
            # else ignore bad filter

    plan["filters"] = sanitized_filters

    # Normalize plan (dedupe selects, remove nested agg, ensure group_by included)
    plan = normalize_plan(plan)

    return plan
