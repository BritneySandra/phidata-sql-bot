# agent.py — FINAL dynamic + FY + Clean SQL plan generation
import os
import json
import re
from datetime import datetime, timedelta
from math import ceil
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"


# ------------------------------------------------------------
# Load helpers
# ------------------------------------------------------------
def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


METADATA = load_json_file("metadata.json")
METRICS = load_json_file("metrics.json")


# ------------------------------------------------------------
# SQL schema
# ------------------------------------------------------------
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
        cursor.execute(
            f"""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{TABLE_NAME}'
            """
        )
        schema = {row.COLUMN_NAME: row.DATA_TYPE.lower() for row in cursor.fetchall()}
        conn.close()
        return schema
    except:
        return {}


_SCHEMA = {}


def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA


# ------------------------------------------------------------
# Groq LLM
# ------------------------------------------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


def choose_best_groq_model(client):
    preferred = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "llama-3.1-8b-instant"
    ]
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        for m in preferred:
            if m in available:
                return m
        for m in available:
            if "llama" in m or "qwen" in m:
                return m
        return available[0]
    except:
        return "llama-3.1-8b-instant"


# ------------------------------------------------------------
# Financial year utilities (FY starts in March)
# ------------------------------------------------------------
def calendar_to_fy_year(year, month):
    return year if month >= 3 else year - 1


def calendar_to_fy_month(m):
    return ((m - 3) % 12) + 1


def fy_quarter(fy_month):
    return ceil(fy_month / 3)


def last_financial_quarter(ref=None):
    if not ref:
        ref = datetime.utcnow()
    fy = calendar_to_fy_year(ref.year, ref.month)
    fm = calendar_to_fy_month(ref.month)
    fq = fy_quarter(fm)
    if fq == 1:
        return fy - 1, 4
    return fy, fq - 1


def previous_calendar_month(ref=None):
    if not ref:
        ref = datetime.utcnow()
    first = ref.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.year, prev.month


def previous_calendar_year(ref=None):
    if not ref:
        ref = datetime.utcnow()
    return ref.year - 1


# ------------------------------------------------------------
# Sanitize filter values: ALWAYS return primitive int/str
# ------------------------------------------------------------
def sanitize_filter_value(v):
    if isinstance(v, (int, float, str)):
        return v

    if isinstance(v, dict):
        for k in ["value", "year", "month", "quarter"]:
            if k in v:
                return sanitize_filter_value(v[k])
        for x in v.values():
            cleaned = sanitize_filter_value(x)
            if cleaned is not None:
                return cleaned
        return None

    if isinstance(v, list) and v:
        return sanitize_filter_value(v[0])

    return None


# ------------------------------------------------------------
# Time parser
# ------------------------------------------------------------
def parse_time_filters(text):
    q = text.lower()
    filters = []

    # FY explicit
    m = re.search(r"\bfy\s*(20\d{2})\b", q)
    if m:
        fy = int(m.group(1))
        filters.append({"column": "FinancialYear", "operator": "=", "value": fy})
        return filters

    # Quarter explicit
    m = re.search(r"(?:q|quarter)\s*([1-4])(?:[^0-9]+(20\d{2}))?", q)
    if m:
        qnum = int(m.group(1))
        yr = int(m.group(2)) if m.group(2) else None
        filters.append({"column": "FinancialQuarter", "operator": "=", "value": qnum})
        if yr:
            filters.append({"column": "FinancialYear", "operator": "=", "value": yr})
        return filters

    # Month + Year
    m = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\w+)\s+(20\d{2})\b",
        q)
    if m:
        mon_name = m.group(1)
        yr = int(m.group(2))
        try:
            cal_m = int(datetime.strptime(mon_name[:3], "%b").month)
        except:
            return []
        fm = calendar_to_fy_month(cal_m)
        fy = calendar_to_fy_year(yr, cal_m)
        filters.append({"column": "FinancialMonth", "operator": "=", "value": fm})
        filters.append({"column": "FinancialYear", "operator": "=", "value": fy})
        return filters

    # last month
    if "last month" in q or "previous month" in q:
        y, m = previous_calendar_month()
        fm = calendar_to_fy_month(m)
        fy = calendar_to_fy_year(y, m)
        filters.append({"column": "FinancialMonth", "operator": "=", "value": fm})
        filters.append({"column": "FinancialYear", "operator": "=", "value": fy})
        return filters

    # last quarter
    if "last quarter" in q or "previous quarter" in q:
        fy, fq = last_financial_quarter()
        filters.append({"column": "FinancialQuarter", "operator": "=", "value": fq})
        filters.append({"column": "FinancialYear", "operator": "=", "value": fy})
        return filters

    # last year (Financial)
    if "last year" in q or "previous year" in q:
        y = previous_calendar_year()
        filters.append({"column": "FinancialYear", "operator": "=", "value": y})
        return filters

    # plain year
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        filters.append({"column": "FinancialYear", "operator": "=", "value": int(m.group(1))})
        return filters

    return filters


# ------------------------------------------------------------
# Clean SELECT and ORDER BY
# ------------------------------------------------------------
AGG = re.compile(r"^\s*(sum|avg|min|max|count)\s*\(", re.I)


def is_agg(expr):
    return isinstance(expr, str) and AGG.match(expr)


def normalize_plan(plan):
    plan = dict(plan)

    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []

    schema = get_schema()

    # Clean group by
    clean_gb = []
    for g in group_by:
        if g in schema and g not in clean_gb:
            clean_gb.append(g)

    # Build cleaned selects
    cleaned = []
    seen = set()

    # Always include group_by
    for g in clean_gb:
        cleaned.append({
            "column": g,
            "expression": None,
            "aggregation": None,
            "alias": g
        })
        seen.add(g)

    for s in selects:
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or col

        if alias in (None, "", "None"):
            continue

        if alias in seen:
            continue

        if expr and is_agg(expr):
            agg = None  # prevent nested SUM(SUM())

        cleaned.append({
            "column": col,
            "expression": expr,
            "aggregation": agg,
            "alias": alias
        })
        seen.add(alias)

    # Order by cleanup
    valid_alias = {x["alias"] for x in cleaned}
    clean_order = []
    for ob in order_by:
        col = ob.get("column")
        if col in valid_alias:
            clean_order.append(ob)

    plan["select"] = cleaned
    plan["group_by"] = clean_gb
    plan["order_by"] = clean_order

    return plan


# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------
def build_prompt(question, schema, metadata, metrics):
    mapping_rules = """
Dimension rules:
- "department" → DeptCode
- "branch" → BranchCode
- "company" → CompanyCode
- "customer" → CustomerLeadGroup
Never use PK columns.
"""

    fy_rules = """
Financial Year:
- FY starts in March and ends in February.
- When user says a plain year (e.g., 2024) → Interpret as FinancialYear.
- Use FinancialMonth/Quarter/Year unless user explicitly says TRANSACTION.
"""

    return f"""
You are a SQL Planning Engine. Return ONLY JSON.

User question:
{question}

Use these rules:
{mapping_rules}
{fy_rules}

Schema:
{json.dumps(schema, indent=2)}

Metadata:
{json.dumps(metadata, indent=2)}

Business Metrics:
{json.dumps(metrics, indent=2)}

OUTPUT FORMAT (JSON only):
{{
 "select": [...],
 "filters": [...],
 "group_by": [...],
 "order_by": [...],
 "limit": null or number
}}
"""


# ------------------------------------------------------------
# Extract + clean plan
# ------------------------------------------------------------
def extract_query(question):
    client = get_client()
    if not client:
        return {"error": "Missing GROQ_API_KEY"}

    schema = get_schema()
    model = choose_best_groq_model(client)
    prompt = build_prompt(question, schema, METADATA, METRICS)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"error": "LLM did not return JSON", "raw": raw}

    text = raw[start:end + 1]

    try:
        plan = json.loads(text)
    except Exception as e:
        return {"error": f"JSON parse error: {e}", "raw": raw}

    # Merge time filters and sanitize
    tf = parse_time_filters(question)
    existing = plan.get("filters", []) or []

    for f in tf:
        f["value"] = sanitize_filter_value(f.get("value"))
        if f["value"] is None:
            continue
        if not any(x.get("column") == f["column"] for x in existing):
            existing.append(f)

    # sanitize all
    for f in existing:
        f["value"] = sanitize_filter_value(f.get("value"))

    plan["filters"] = existing

    # Normalize everything
    plan = normalize_plan(plan)
    return plan
