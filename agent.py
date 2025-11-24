# agent.py
from groq import Groq
import os
import json
import pyodbc
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"


# ---------------------------------------------------------
# LOAD SQL SCHEMA
# ---------------------------------------------------------
def load_sql_schema():
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            f"Encrypt=no;"
            f"TrustServerCertificate=yes;",
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

    except Exception as e:
        print("⚠ Schema load failed:", e)
        return {}


_SCHEMA = {}


def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA


def numeric_columns():
    schema = get_schema()
    return [
        c for c, t in schema.items()
        if t in ("decimal", "numeric", "money", "float", "int", "bigint", "smallint")
    ]


def categorical_columns():
    schema = get_schema()
    nums = set(numeric_columns())
    return [
        c for c, t in schema.items()
        if c not in nums and t not in ("date", "datetime", "smalldatetime", "datetime2")
    ]


# ---------------------------------------------------------
# BUSINESS METRICS + RULES
# ---------------------------------------------------------
BUSINESS_METRICS = {
    "revenue": {
        "keywords": ["revenue", "total revenue", "sales", "turnover", "income"],
        "expression": "[REVAmount] + [WIPAmount]",
        "base_column": "REVAmount",
        "default_agg": "sum",
        "alias": "total_revenue"
    },
    "cost": {
        "keywords": ["cost", "expense", "total cost"],
        "expression": "[CSTAmount] + [ACRAmount]",
        "base_column": "CSTAmount",
        "default_agg": "sum",
        "alias": "total_cost"
    },
    "profit": {
        "keywords": ["profit", "margin", "jobprofit"],
        "expression": "[JobProfit]",
        "base_column": "JobProfit",
        "default_agg": "sum",
        "alias": "total_profit"
    },
}

BUSINESS_RULES_TEXT = """
Key business rules:
- Revenue = REVAmount + WIPAmount
- Cost = CSTAmount + ACRAmount
- Profit = JobProfit
"""

COLUMN_DESCRIPTIONS_TEXT = """
Column descriptions include job identifiers, fiscal time, transport mode,
customer group, product levels, cost & revenue fields, volume fields, etc.
"""


# ---------------------------------------------------------
# TIME PARSING
# ---------------------------------------------------------
def parse_time_from_text(question: str):
    q = question.lower()
    now = datetime.utcnow()
    out = {"year": None, "quarter": None, "month": None}

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        out["year"] = int(m.group(1))

    if "last year" in q or "previous year" in q:
        out["year"] = now.year - 1

    if "last quarter" in q or "previous quarter" in q:
        current_q = (now.month - 1) // 3 + 1
        prev_q = current_q - 1
        prev_year = now.year
        if prev_q == 0:
            prev_q = 4
            prev_year -= 1
        out["quarter"] = prev_q
        out["year"] = prev_year

    m = re.search(r"(?:q|quarter)[^\d]*([1-4])(?:[^0-9]+(20\d{2}))?", q)
    if m:
        out["quarter"] = int(m.group(1))
        if m.group(2):
            out["year"] = int(m.group(2))

    if "last month" in q or "previous month" in q:
        prev = now.replace(day=1) - timedelta(days=1)
        out["month"] = prev.month
        out["year"] = prev.year

    months = {
        name.lower(): idx
        for idx, name in enumerate([
            "", "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ])
    }
    for name, idx in months.items():
        if name and name in q:
            out["month"] = idx
            if not out["year"]:
                out["year"] = now.year

    return out


# ---------------------------------------------------------
# DETECTION HELPERS
# ---------------------------------------------------------
def detect_top_n(q):
    q = q.lower()
    m = re.search(r"top\s+(\d+)", q)
    if m: return int(m.group(1))
    m = re.search(r"first\s+(\d+)", q)
    if m: return int(m.group(1))
    return None


def detect_business_metric_key(q):
    q = q.lower()
    for key, meta in BUSINESS_METRICS.items():
        for kw in meta["keywords"]:
            if kw in q:
                return key
    return None


def detect_metric_column(q):
    nums = numeric_columns()
    q = q.lower()
    for col in nums:
        if col.lower() in q:
            return col
    return nums[0] if nums else None


def detect_group_column(q):
    q = q.lower()
    cats = categorical_columns()

    if " by " in q:
        phrase = q.split(" by ")[1].strip()
        for col in cats:
            if phrase in col.lower():
                return col

    return None


# ---------------------------------------------------------
# LLM CLIENT
# ---------------------------------------------------------
def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)


# ---------------------------------------------------------
# MAIN QUERY PARSER (THIS GENERATES PLAN)
# ---------------------------------------------------------
def extract_query(question: str):
    schema = get_schema()
    q_lower = question.lower()
    nums = numeric_columns()
    cats = categorical_columns()

    time_ctx = parse_time_from_text(question)
    top_n = detect_top_n(question)
    metric_key = detect_business_metric_key(question)

    if metric_key:
        bm = BUSINESS_METRICS[metric_key]
        select_entry = {
            "column": bm["base_column"],
            "expression": bm["expression"],
            "aggregation": bm["default_agg"],
            "alias": bm["alias"]
        }
    else:
        col = detect_metric_column(question)
        select_entry = {
            "column": col,
            "expression": None,
            "aggregation": "sum",
            "alias": col
        }

    # ❗ DO NOT GROUP unless user explicitly says "by ..."
    group_col = None
    if " by " in q_lower or " breakdown" in q_lower or " group " in q_lower:
        group_col = detect_group_column(question)

    plan = {
        "select": [select_entry],
        "filters": [],
        "group_by": [group_col] if group_col else [],
        "order_by": [],
        "limit": top_n
    }

    # Add correct time filters
    if time_ctx.get("year"):
        plan["filters"].append({
            "column": "FinancialYear", "operator": "=", "value": time_ctx["year"]
        })
    if time_ctx.get("quarter"):
        plan["filters"].append({
            "column": "FinancialQuarter", "operator": "=", "value": time_ctx["quarter"]
        })
    if time_ctx.get("month"):
        plan["filters"].append({
            "column": "FinancialMonth", "operator": "=", "value": time_ctx["month"]
        })

    return plan
