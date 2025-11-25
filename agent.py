# agent.py — FINAL dynamic agent with FY logic + safe filters + dimension mapping
import os
import json
import re
from datetime import datetime, timedelta
from math import ceil
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"

# --------------------------------------
# Load metadata and metrics JSON
# --------------------------------------
def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

METADATA = load_json_file("metadata.json")
METRICS = load_json_file("metrics.json")


# --------------------------------------
# SQL SCHEMA LOADER
# --------------------------------------
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
        result = {row.COLUMN_NAME: row.DATA_TYPE.lower() for row in cursor.fetchall()}
        conn.close()
        return result
    except Exception as e:
        print("⚠ Could not load schema:", e)
        return {}

_SCHEMA = {}
def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA


# ----------------------------------------------------------
# GROQ MODEL
# ----------------------------------------------------------
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
        avail = [m.id for m in models.data]
        for m in preferred:
            if m in avail:
                return m
        return avail[0] if avail else "llama-3.1-8b-instant"
    except:
        return "llama-3.1-8b-instant"


# ----------------------------------------------------------
# FINANCIAL YEAR LOGIC (FY starts March)
# ----------------------------------------------------------
def current_utc():
    return datetime.utcnow()

def calendar_to_fy_year(year, month):
    return year if month >= 3 else year - 1

def calendar_to_fy_month(calendar_month):
    return ((calendar_month - 3) % 12) + 1  # FY month 1=Mar, 12=Feb

def fy_quarter_from_fy_month(fy_month):
    return ceil(fy_month / 3)

def previous_calendar_month(reference=None):
    if not reference:
        reference = current_utc()
    first = reference.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.year, prev.month

def previous_calendar_year(reference=None):
    return (reference or current_utc()).year - 1

def last_financial_quarter(reference=None):
    if not reference:
        reference = current_utc()

    fy_year = calendar_to_fy_year(reference.year, reference.month)
    fy_month = calendar_to_fy_month(reference.month)
    fq = fy_quarter_from_fy_month(fy_month)

    if fq == 1:
        return fy_year - 1, 4
    return fy_year, fq - 1


# ----------------------------------------------------------
# TIME FILTER PARSER
# ----------------------------------------------------------
def parse_time_filters(text):
    q = text.lower()
    filters = []

    # FY explicitly
    m = re.search(r'\b(?:fy|financial year)\s*(20\d{2})\b', q)
    if m:
        fy = int(m.group(1))
        filters.append({"column": "FinancialYear", "operator": "=", "value": fy})
        return filters

    # LAST / PREVIOUS QUARTER
    if "last quarter" in q or "previous quarter" in q:
        fy, fq = last_financial_quarter()
        filters.append({"column": "FinancialQuarter", "operator": "=", "value": fq})
        filters.append({"column": "FinancialYear", "operator": "=", "value": fy})
        return filters

    # LAST / PREVIOUS MONTH
    if "last month" in q or "previous month" in q:
        y, m = previous_calendar_month()
        fy = calendar_to_fy_year(y, m)
        fm = calendar_to_fy_month(m)
        filters.append({"column": "FinancialMonth", "operator": "=", "value": fm})
        filters.append({"column": "FinancialYear", "operator": "=", "value": fy})
        return filters

    # LAST / PREVIOUS YEAR → FY
    if "last year" in q or "previous year" in q:
        prev = previous_calendar_year()
        filters.append({"column": "FinancialYear", "operator": "=", "value": prev})
        return filters

    # YEAR mentioned → default to FinancialYear
    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        yr = int(m.group(1))
        filters.append({"column": "FinancialYear", "operator": "=", "value": yr})
        return filters

    return filters


# ----------------------------------------------------------
# DIMENSION MAPPING (YOUR RULE)
# ----------------------------------------------------------
DIMENSION_MAP = {
    "department": "DeptCode",
    "dept": "DeptCode",
    "branch": "BranchCode",
    "company": "CompanyCode",
    "customer": "CustomerLeadGroup",
    "customer group": "CustomerLeadGroup",
}

PK_COLUMNS = {
    "JobPK", "CountryPK", "CompanyPK",
    "BranchPK", "DeptPK"
}


# ----------------------------------------------------------
# SAFE PLAN NORMALIZER
# ----------------------------------------------------------
AGG_RE = re.compile(r"^\s*(sum|avg|min|max|count)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr):
    return isinstance(expr, str) and AGG_RE.match(expr.strip())


def normalize_plan(plan):
    schema = get_schema()

    selects = plan.get("select", [])
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []

    # --------------------------------
    # 1. Apply dimension mapping
    # --------------------------------
    mapped_group_by = []
    for g in group_by:
        if g.lower() in DIMENSION_MAP:
            mapped_group_by.append(DIMENSION_MAP[g.lower()])
        elif g not in PK_COLUMNS and g in schema:
            mapped_group_by.append(g)

    group_by = list(dict.fromkeys(mapped_group_by))


    # --------------------------------
    # 2. Build SELECT cleanly (no duplicates)
    # --------------------------------
    clean_selects = []
    seen = set()

    # group_by columns must appear in SELECT once
    for g in group_by:
        clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})
        seen.add(g)

    for s in selects:
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or (col or "value")

        # dimension alias mapping
        if alias.lower() in DIMENSION_MAP:
            alias = DIMENSION_MAP[alias.lower()]
            col = alias

        # skip PKs
        if col in PK_COLUMNS:
            continue

        if alias in seen:
            continue
        seen.add(alias)

        if isinstance(expr, str) and is_aggregate_expression(expr):
            agg = None

        clean_selects.append({
            "column": col,
            "expression": expr,
            "aggregation": agg,
            "alias": alias
        })

    plan["select"] = clean_selects
    plan["group_by"] = group_by

    # order_by only if alias exists
    select_aliases = {s["alias"] for s in clean_selects}
    plan["order_by"] = [
        ob for ob in order_by if ob.get("column") in select_aliases
    ]

    return plan


# ----------------------------------------------------------
# PROMPT FOR LLM
# ----------------------------------------------------------
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- FY starts in March and ends in February.\n"
        "- Use FinancialYear, FinancialQuarter, FinancialMonth for all time unless user says TRANSACTION explicitly.\n"
        "- NEVER return objects like {\"function\":..., \"args\":...}; always return literal integers.\n"
        "- department → DeptCode, branch → BranchCode, company → CompanyCode, customer → CustomerLeadGroup.\n"
        "- Never use PK columns.\n"
        "- Output STRICT JSON: select, filters, group_by, order_by, limit.\n"
    )

    return f"""
You are a SQL planning engine. Convert the user's question into STRICT JSON (no explanation).

{rules}

USER QUESTION:
{question}

TABLE: {TABLE_NAME}
SCHEMA: {json.dumps(schema, indent=2)}
COLUMN DESCRIPTIONS: {json.dumps(metadata, indent=2)}
METRICS: {json.dumps(metrics, indent=2)}

Return JSON only.
"""


# ----------------------------------------------------------
# MAIN LLM QUERY PLANNER
# ----------------------------------------------------------
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
    if start < 0 or end < 0:
        return {"error": "Invalid JSON", "raw": raw}

    plan = json.loads(raw[start:end+1])

    # Local time filters
    time_filters = parse_time_filters(question)

    # Merge time & LLM filters
    merged = plan.get("filters", [])
    for tf in time_filters:
        if not any(f["column"] == tf["column"] for f in merged):
            merged.append(tf)
    plan["filters"] = merged

    # ------------------------------------------------------
    # FINAL SANITIZATION — REMOVE dict filter values
    # ------------------------------------------------------
    clean = []
    for f in plan["filters"]:
        val = f["value"]

        if isinstance(val, dict):     # LLM hallucination
            q_lower = question.lower()

            if "last quarter" in q_lower or "previous quarter" in q_lower:
                fy, fq = last_financial_quarter()
                if f["column"] == "FinancialQuarter": f["value"] = fq
                if f["column"] == "FinancialYear": f["value"] = fy

            elif "last month" in q_lower or "previous month" in q_lower:
                y, m = previous_calendar_month()
                fy = calendar_to_fy_year(y, m)
                fm = calendar_to_fy_month(m)
                if f["column"] == "FinancialMonth": f["value"] = fm
                if f["column"] == "FinancialYear": f["value"] = fy

            elif "last year" in q_lower or "previous year" in q_lower:
                prev = previous_calendar_year()
                if f["column"] == "FinancialYear": f["value"] = prev
        clean.append(f)

    plan["filters"] = clean

    return normalize_plan(plan)
