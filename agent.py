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

# --------------------------------
# Load schema from SQL Server
# --------------------------------
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
        print("⚠ SQL schema load failed:", e)
        return {}

_SCHEMA = {}

def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA


def numeric_columns():
    schema = get_schema()
    return [c for c, t in schema.items()
            if t in ('decimal', 'numeric', 'money', 'float', 'int', 'bigint', 'smallint')]


def categorical_columns():
    schema = get_schema()
    nums = set(numeric_columns())
    return [
        c for c, t in schema.items()
        if c not in nums and t not in ('date', 'datetime', 'smalldatetime', 'datetime2')
    ]


# --------------------------------
# Business metric rules
# --------------------------------
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
        "keywords": ["profit", "margin"],
        "expression": "[JobProfit]",
        "base_column": "JobProfit",
        "default_agg": "sum",
        "alias": "total_profit"
    }
}

BUSINESS_RULES_TEXT = """
Revenue = REVAmount + WIPAmount
Cost = CSTAmount + ACRAmount
Profit = JobProfit
"""

COLUMN_DESCRIPTIONS_TEXT = """
TransportMode, ProductLevel, Customer, Country fields...
"""


# --------------------------------
# Updated Time Parsing (Fiscal Month March→Feb)
# --------------------------------
def parse_time_from_text(question: str):
    q = question.lower()
    now = datetime.utcnow()

    res = {"year": None, "quarter": None, "month": None, "timeframe": None}

    # Detect explicit year
    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        res["year"] = int(m.group(1))

    # Last year
    if "last year" in q or "previous year" in q:
        res["year"] = now.year - 1
        res["timeframe"] = "previous_year"

    # Last quarter
    if "last quarter" in q or "previous quarter" in q:
        current_q = (now.month - 1)//3 + 1
        prev_q = current_q - 1
        prev_year = now.year
        if prev_q == 0:
            prev_q = 4
            prev_year -= 1
        res["quarter"] = prev_q
        res["year"] = prev_year

    # Explicit quarter
    m = re.search(r'(?:q|quarter)[^\d]*([1-4])', q)
    if m:
        res["quarter"] = int(m.group(1))

    # Last month
    if "last month" in q or "previous month" in q:
        prev = now.replace(day=1) - timedelta(days=1)
        res["month"] = prev.month
        res["year"] = prev.year

    # 🔥 Fiscal Year Month Mapping (NEW)
    fiscal_month_map = {
        "march": 1,
        "april": 2,
        "may": 3,
        "june": 4,
        "july": 5,
        "august": 6,
        "september": 7,
        "october": 8,
        "november": 9,
        "december": 10,
        "january": 11,
        "february": 12
    }

    for name, idx in fiscal_month_map.items():
        if name in q:
            res["month"] = idx
            if not res["year"]:
                res["year"] = now.year
            break

    return res


# --------------------------------
# Top N detection
# --------------------------------
def detect_top_n(question: str):
    q = question.lower()
    m = re.search(r'top\s+(\d+)', q)
    if m: return int(m.group(1))
    m = re.search(r'first\s+(\d+)', q)
    if m: return int(m.group(1))
    return None


def detect_business_metric_key(question: str):
    q = question.lower()
    for key, meta in BUSINESS_METRICS.items():
        for kw in meta["keywords"]:
            if kw in q:
                return key
    return None


def detect_metric_column(question: str):
    q = question.lower()
    nums = numeric_columns()
    schema = get_schema()

    for col in nums:
        if col.lower() in q:
            return col

    if "profit" in q and "JobProfit" in schema:
        return "JobProfit"

    if ("revenue" in q or "sales" in q) and "REVAmount" in schema:
        return "REVAmount"

    return nums[0] if nums else None


def detect_group_column(question: str):
    q = question.lower()
    cats = categorical_columns()

    m = re.search(r'by\s+([a-z0-9 _-]+)', q)
    if m:
        cand = m.group(1).strip().replace(" ", "").lower()
        for col in cats:
            if cand in col.lower():
                return col

    if "transport" in q and "TransportMode" in cats:
        return "TransportMode"

    if "customer" in q and "CustomerName" in cats:
        return "CustomerName"

    return None


# --------------------------------
# Groq Client
# --------------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# --------------------------------
# Build query plan
# --------------------------------
def extract_query(question: str):
    schema = get_schema()
    nums = numeric_columns()
    cats = categorical_columns()
    time_ctx = parse_time_from_text(question)
    q_lower = question.lower()

    # Business Metric?
    bm_key = detect_business_metric_key(question)
    bm_meta = BUSINESS_METRICS.get(bm_key) if bm_key else None

    agg = "sum"
    if "avg" in q_lower: agg = "avg"
    if "count" in q_lower: agg = "count"

    if bm_meta:
        select_entry = {
            "column": bm_meta["base_column"],
            "expression": bm_meta["expression"],
            "aggregation": bm_meta["default_agg"],
            "alias": bm_meta["alias"],
            "metric_key": bm_key
        }
    else:
        metric_col = detect_metric_column(question)
        select_entry = {
            "column": metric_col,
            "expression": None,
            "aggregation": agg,
            "alias": metric_col,
            "metric_key": None
        }

    group_col = detect_group_column(question)
    top_n = detect_top_n(question)

    plan = {
        "select": [select_entry],
        "filters": [],
        "group_by": [group_col] if group_col else [],
        "order_by": [],
        "limit": top_n
    }

    # Inject Time Filters
    filters = []
    if time_ctx.get("year"):
        filters.append({"column": "FinancialYear", "operator": "=", "value": time_ctx["year"]})
    if time_ctx.get("quarter"):
        filters.append({"column": "FinancialQuarter", "operator": "=", "value": time_ctx["quarter"]})
    if time_ctx.get("month"):
        filters.append({"column": "FinancialMonth", "operator": "=", "value": time_ctx["month"]})

    plan["filters"] = filters

    return plan
