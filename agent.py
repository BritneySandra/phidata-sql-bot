# ---------------------------
# agent.py (FINAL – QWEN FIX)
# ---------------------------

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
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

METADATA = load_json_file("metadata.json")
METRICS = load_json_file("metrics.json")


# -------------------------
# Load SQL Schema
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
            timeout=5,
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
# Groq client
# -------------------------
def get_client():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


# FORCE qwen/qwen3-32b (stable & available)
def choose_best_groq_model(client):
    return "qwen/qwen3-32b"


# -------------------------
# Time utility functions
# -------------------------
def calendar_to_fy_year(year, month):
    return year if month >= 3 else year - 1

def month_name_to_num(name):
    try:
        return datetime.strptime(name[:3], "%b").month
    except:
        try:
            return datetime.strptime(name, "%B").month
        except:
            return None

def calendar_to_fy_month(calendar_month):
    return ((calendar_month - 3) % 12) + 1


# -------------------------
# Time Filter Parser
# -------------------------
def parse_time_filters(text):
    q = text.lower()
    filters = []

    # FY 2024
    m = re.search(r"(?:fy|financial year)\s*(20\d{2})", q)
    if m:
        filters.append({"column": "FinancialYear", "operator": "=", "value": int(m.group(1))})
        return filters

    # Jan 2024
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20\d{2})", q)
    if m:
        mon = month_name_to_num(m.group(1))
        yr = int(m.group(2))
        filters.append({"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(mon)})
        filters.append({"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(yr, mon)})
        return filters

    # plain year 2024
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        filters.append({"column": "FinancialYear", "operator": "=", "value": int(m.group(1))})
        return filters

    return filters


# -------------------------
# Plan Normalization Helpers
# -------------------------
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr):
    if not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None


def normalize_plan(plan):
    if not isinstance(plan, dict):
        return {"select": [], "filters": [], "group_by": [], "order_by": []}

    selects = plan.get("select", []) or []
    filters = plan.get("filters", []) or []
    group_by = plan.get("group_by", []) or []

    clean_selects = []
    used = set()

    # Prevent duplicates & nested aggs
    for s in selects:
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = (s.get("alias") or col or "value").replace("%", "_pct")

        if alias in used:
            continue
        used.add(alias)

        if expr and is_aggregate_expression(expr):
            agg = None

        clean_selects.append({
            "column": col,
            "expression": expr,
            "aggregation": agg,
            "alias": alias
        })

    plan["select"] = clean_selects
    plan["filters"] = filters
    plan["group_by"] = group_by
    return plan


# -------------------------
# Prompt Builder
# -------------------------
def build_prompt(question, schema, metadata, metrics):

    return f"""
Return ONLY a VALID JSON object. NO text, NO markdown, NO commentary.

USER QUESTION:
{question}

TABLE: {TABLE_NAME}
SCHEMA: {json.dumps(schema)}
METADATA: {json.dumps(metadata)}
METRICS: {json.dumps(metrics)}

Format exactly like this:

{{
  "select": [
      {{"column": "REVAmount", "expression": null, "aggregation": "SUM", "alias": "total_revenue"}}
  ],
  "filters": [],
  "group_by": [],
  "order_by": []
}}

Return ONLY JSON.
"""


# -------------------------
# STRICT JSON Extractor
# -------------------------
def extract_query(question: str):
    client = get_client()
    if not client:
        return {"error": "Missing GROQ_API_KEY"}

    schema = get_schema()
    model = choose_best_groq_model(client)
    prompt = build_prompt(question, schema, METADATA, METRICS)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = resp.choices[0].message.content.strip()

    # extract pure JSON (handles markdown, text, explanations)
    json_blocks = re.findall(r"{(?:[^{}]|(?:\{[^{}]*\}))*}", raw, re.DOTALL)

    if not json_blocks:
        return {"error": "json_parse_failed", "raw": raw}

    plan = None
    for blk in json_blocks:
        try:
            plan = json.loads(blk)
            break
        except:
            continue

    if not isinstance(plan, dict):
        return {"error": "json_parse_failed", "raw": raw}

    # append time filters
    time_filters = parse_time_filters(question)
    existing = plan.get("filters", [])
    plan["filters"] = existing + time_filters

    return normalize_plan(plan)
