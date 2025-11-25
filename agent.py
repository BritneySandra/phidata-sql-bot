# ===========================
# agent.py (FINAL UPDATED)
# ===========================
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
# Load metadata
# -------------------------
def load_json_file(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return {}

METADATA = load_json_file("metadata.json")
METRICS = load_json_file("metrics.json")

# -------------------------
# Load SQL schema
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
        print("⚠ Schema load failed:", e)
        return {}

_SCHEMA = {}

def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA

# -------------------------
# Groq client
# -------------------------
def get_client():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None

def choose_best_groq_model(client):
    """Force using qwen/qwen3-32b to avoid rate-limit issues."""
    try:
        models = client.models.list()
        available = [m.id for m in models.data]

        if "qwen/qwen3-32b" in available:
            return "qwen/qwen3-32b"

        # fallback: pick ANY smaller llama
        for m in available:
            if "llama" in m and "70b" not in m.lower():
                return m

        return available[0] if available else "qwen/qwen3-32b"

    except:
        return "qwen/qwen3-32b"

# -------------------------
# TIME UTILITIES
# -------------------------
def calendar_to_fy_year(y,m): return y if m>=3 else y-1
def calendar_to_fy_month(m): return ((m - 3) % 12) + 1
def month_name_to_num(name):
    try:
        return datetime.strptime(name[:3].capitalize(), "%b").month
    except:
        return None

def previous_calendar_month(ref=None):
    if not ref: ref = datetime.utcnow()
    first = ref.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.year, prev.month

def previous_calendar_year(ref=None):
    if not ref: ref = datetime.utcnow()
    return ref.year - 1

def fy_quarter_from_fy_month(m): return ceil(m / 3)

def last_financial_quarter(ref=None):
    if not ref: ref = datetime.utcnow()
    fy = calendar_to_fy_year(ref.year, ref.month)
    fm = calendar_to_fy_month(ref.month)
    fq = fy_quarter_from_fy_month(fm)
    if fq == 1:
        return fy-1, 4
    return fy, fq-1

# -------------------------
# Time phrase parser
# -------------------------
def parse_time_filters(q: str):
    q = q.lower()
    filters = []

    # FY 2024
    m = re.search(r'(?:fy|financial year)\s*(20\d{2})', q)
    if m:
        y = int(m.group(1))
        filters.append({"column": "FinancialYear", "operator": "=", "value": y})
        return filters

    # Quarter
    m = re.search(r'(?:q|quarter)\s*([1-4])(?:\s*(20\d{2}))?', q)
    if m:
        qnum = int(m.group(1))
        y = int(m.group(2)) if m.group(2) else None
        filters.append({"column": "FinancialQuarter", "operator": "=", "value": qnum})
        if y: filters.append({"column": "FinancialYear", "operator": "=", "value": y})
        return filters

    # Month + Year
    m = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(20\d{2})', q)
    if m:
        month = month_name_to_num(m.group(1))
        year = int(m.group(2))
        fy = calendar_to_fy_year(year, month)
        fm = calendar_to_fy_month(month)
        filters.append({"column": "FinancialMonth", "operator": "=", "value": fm})
        filters.append({"column": "FinancialYear", "operator": "=", "value": fy})
        return filters

    # last month
    if "last month" in q or "previous month" in q:
        y, m = previous_calendar_month()
        filters.append({"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(y,m)})
        filters.append({"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(m)})
        return filters

    # last year
    if "last year" in q or "previous year" in q:
        filters.append({"column": "FinancialYear", "operator": "=", "value": previous_calendar_year()})
        return filters

    # plain year
    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        filters.append({"column": "FinancialYear", "operator": "=", "value": int(m.group(1))})
        return filters

    return filters

# -------------------------
# CLEAN & NORMALIZE PLAN
# -------------------------
AGG_RE = re.compile(r"(sum|avg|count|min|max)\(", re.IGNORECASE)

def is_aggregate(expr):
    if not expr or not isinstance(expr,str): return False
    return bool(AGG_RE.search(expr))

def normalize_plan(plan):
    if not isinstance(plan, dict): return plan

    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    schema = get_schema()

    clean_selects = []
    used = set()

    # always ensure group_by columns in select
    for g in group_by:
        if g in schema:
            clean_selects.append({"column": g, "alias": g, "aggregation": None, "expression": None})
            used.add(g)

    for s in selects:
        alias = s.get("alias") or s.get("column") or "value"
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")

        alias = alias.replace("%","pct")

        if alias in used: continue
        used.add(alias)

        # avoid double aggregates
        if expr and is_aggregate(expr):
            agg = None

        clean_selects.append({"column": col, "expression": expr, "aggregation": agg, "alias": alias})

    plan["select"] = clean_selects
    return plan

# -------------------------
# LLM Prompt
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    return f"""
Convert the user question into a STRICT JSON SQL plan.
NO explanation. ONLY JSON.

User Question:
{question}

Table: {TABLE_NAME}

Schema:
{json.dumps(schema)}

Column Descriptions:
{json.dumps(metadata)}

Business Metrics:
{json.dumps(metrics)}

JSON Keys Required:
select, filters, group_by, order_by, limit
"""

# -------------------------
# Extract query
# -------------------------
def extract_query(question: str):
    client = get_client()
    if not client:
        return {"error": "Missing GROQ_API_KEY"}

    schema = get_schema()
    prompt = build_prompt(question, schema, METADATA, METRICS)
    model = choose_best_groq_model(client)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    # extract JSON
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"error":"Invalid LLM output", "raw": raw}

    raw_json = raw[start:end+1].replace("%", "pct")

    try:
        plan = json.loads(raw_json)
    except Exception as e:
        return {"error": f"JSON parse failed: {e}", "raw": raw_json}

    # add time filters
    tf = parse_time_filters(question)
    existing = plan.get("filters", []) or []

    for f in tf:
        if not any(x.get("column")==f.get("column") for x in existing):
            existing.append(f)

    plan["filters"] = existing

    # normalize
    return normalize_plan(plan)
