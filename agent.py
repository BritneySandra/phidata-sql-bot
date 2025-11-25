# agent.py — Stable JSON-safe version for Groq + Qwen models
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
    preferred = [
        "qwen/qwen3-32b",  
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant",
    ]
    try:
        models = [m.id for m in client.models.list().data]
        for m in preferred:
            if m in models:
                return m
        for m in models:
            if "qwen" in m or "llama" in m:
                return m
        return models[0] if models else "llama-3.1-8b-instant"
    except:
        return "qwen/qwen3-32b"

# -------------------------
# Time utilities (FY logic)
# -------------------------
def calendar_to_fy_year(year, month):
    return year if month >= 3 else year - 1

def month_name_to_num(name):
    try:
        return datetime.strptime(name[:3].capitalize(), "%b").month
    except:
        try:
            return datetime.strptime(name, "%B").month
        except:
            return None

def calendar_to_fy_month(calendar_month):
    return ((calendar_month - 3) % 12) + 1

def fy_quarter_from_fy_month(fm):
    return ceil(fm / 3)

def last_financial_quarter(reference=None):
    if not reference:
        reference = datetime.utcnow()
    fy_year = calendar_to_fy_year(reference.year, reference.month)
    fy_month = calendar_to_fy_month(reference.month)
    fq = fy_quarter_from_fy_month(fy_month)
    if fq == 1:
        return fy_year - 1, 4
    return fy_year, fq - 1

def previous_calendar_month(reference=None):
    if not reference:
        reference = datetime.utcnow()
    first = reference.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.year, prev.month

def previous_calendar_year(reference=None):
    if not reference:
        reference = datetime.utcnow()
    return reference.year - 1

# -------------------------
# Filter sanitization
# -------------------------
def sanitize_filter_value(val):
    if isinstance(val, dict):
        for k in ("year","quarter","month","value"):
            if k in val: return val[k]
        for v in val.values():
            if isinstance(v, int): return v
        return str(val)
    return val

# -------------------------
# Time parsing
# -------------------------
def parse_time_filters(text):
    q = (text or "").lower()
    filters = []

    m = re.search(r'\b(?:fy|financial year)\s*(20\d{2})\b', q)
    if m:
        fy = int(m.group(1))
        filters.append({"column":"FinancialYear","operator":"=","value":fy})
        return filters

    m = re.search(r'\b(?:q|quarter)\s*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        qnum = int(m.group(1))
        year = m.group(2)
        filters.append({"column":"FinancialQuarter","operator":"=","value":qnum})
        if year:
            filters.append({"column":"FinancialYear","operator":"=","value":int(year)})
        return filters

    m = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
                  r'january|february|march|april|may|june|july|august|september|october|november|december)'
                  r'\s+(20\d{2})', q)
    if m:
        mon = m.group(1)
        yr = int(m.group(2))
        cal_month = month_name_to_num(mon)
        if cal_month:
            filters.append({"column":"FinancialMonth","operator":"=","value":calendar_to_fy_month(cal_month)})
            filters.append({"column":"FinancialYear","operator":"=","value":calendar_to_fy_year(yr, cal_month)})
            return filters

    if "last month" in q or "previous month" in q:
        y, m = previous_calendar_month()
        filters.append({"column":"FinancialMonth","operator":"=","value":calendar_to_fy_month(m)})
        filters.append({"column":"FinancialYear","operator":"=","value":calendar_to_fy_year(y, m)})
        return filters

    if "last quarter" in q or "previous quarter" in q:
        fy, fq = last_financial_quarter()
        filters.append({"column":"FinancialQuarter","operator":"=","value":fq})
        filters.append({"column":"FinancialYear","operator":"=","value":fy})
        return filters

    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        filters.append({"column":"FinancialYear","operator":"=","value":int(m.group(1))})
        return filters

    return filters

# -------------------------
# Plan normalization
# -------------------------
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr):
    return isinstance(expr, str) and AGG_RE.match(expr.strip())

def normalize_plan(plan):
    plan = dict(plan)
    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []

    seen_alias = set()
    clean_selects = []

    schema = get_schema()
    group_clean = [g for g in group_by if g in schema]

    for g in group_clean:
        clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})
        seen_alias.add(g)

    for s in selects:
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = (s.get("alias") or col or "value").replace("%", "_pct")

        if alias in seen_alias:
            continue

        if expr and is_aggregate_expression(expr):
            agg = None
        if agg in ("null","none","NULL","None"):
            agg = None

        clean_selects.append({
            "column": col,
            "expression": expr,
            "aggregation": agg,
            "alias": alias
        })
        seen_alias.add(alias)

    plan["select"] = clean_selects
    plan["group_by"] = group_clean
    plan["order_by"] = order_by
    return plan

# -------------------------
# LLM prompt builder
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    return f"""
You MUST return ONLY JSON. No explanation. No markdown.

USER QUESTION:
{question}

TABLE: {TABLE_NAME}

SCHEMA:
{json.dumps(schema, indent=2)}

COLUMN DESCRIPTIONS:
{json.dumps(metadata, indent=2)}

BUSINESS METRICS:
{json.dumps(metrics, indent=2)}

OUTPUT JSON KEYS:
select, filters, group_by, order_by, limit
"""

# -------------------------
# FIXED SAFE extract_query()
# -------------------------
def extract_query(question: str):
    client = get_client()
    if not client:
        return {"error": "Missing GROQ_API_KEY"}

    schema = get_schema()
    prompt = build_prompt(question, schema, METADATA, METRICS)
    model_name = choose_best_groq_model(client)

    forced_prompt = (
        "Return ONLY valid JSON. No text outside JSON.\n"
        "If cannot answer, return {}.\n\n"
        + prompt
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Output must be VALID JSON only."},
                {"role": "user", "content": forced_prompt}
            ],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()

    except Exception as e:
        return {"error": f"Groq API error: {e}"}

    # Try extracting JSON
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found")

        json_text = raw[start:end+1].replace("%", "_pct")
        plan = json.loads(json_text)

    except Exception:
        # JSON Repair
        try:
            fix_prompt = f"""
Fix the following into VALID JSON only.
--- START ---
{raw}
--- END ---
"""
            repair = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": "Fix malformed JSON. Output ONLY JSON."},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0
            )
            fixed = repair.choices[0].message.content.strip()
            fixed_json = fixed[fixed.find("{"):fixed.rfind("}")+1]
            plan = json.loads(fixed_json)

        except Exception as e2:
            return {"error": "json_parse_failed", "raw": raw, "repair_error": str(e2)}

    # Time filters
    time_filters = parse_time_filters(question)
    plan_filters = plan.get("filters", []) or []
    for tf in time_filters:
        tf["value"] = sanitize_filter_value(tf["value"])
        if not any(f.get("column") == tf["column"] for f in plan_filters):
            plan_filters.append(tf)
    plan["filters"] = plan_filters

    # Normalize
    return normalize_plan(plan)
