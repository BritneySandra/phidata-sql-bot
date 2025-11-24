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


# ------------------------------------------------------
# Load SQL Schema
# ------------------------------------------------------
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
    return [
        c for c, t in schema.items()
        if t in ('decimal', 'numeric', 'money', 'float', 'int', 'bigint', 'smallint')
    ]


def categorical_columns():
    schema = get_schema()
    nums = set(numeric_columns())
    return [
        c for c, t in schema.items()
        if c not in nums and t not in ('date', 'datetime', 'smalldatetime', 'datetime2')
    ]


# ------------------------------------------------------
# Business Metrics (Option A)
# ------------------------------------------------------
BUSINESS_METRICS = {
    "revenue": {
        "keywords": ["revenue", "total revenue", "sales", "turnover", "income"],
        "expression": "[REVAmount] + [WIPAmount]",
        "base_column": "REVAmount",
        "default_agg": "sum",
        "alias": "total_revenue"
    },
    "cost": {
        "keywords": ["cost", "total cost", "expense"],
        "expression": "[CSTAmount] + [ACRAmount]",
        "base_column": "CSTAmount",
        "default_agg": "sum",
        "alias": "total_cost"
    },
    "profit": {
        "keywords": ["profit", "jobprofit", "margin"],
        "expression": "[JobProfit]",
        "base_column": "JobProfit",
        "default_agg": "sum",
        "alias": "total_profit"
    }
}


# ------------------------------------------------------
# Time Parsing
# ------------------------------------------------------
def parse_time_from_text(question: str):
    q = question.lower()
    now = datetime.utcnow()

    res = {"year": None, "quarter": None, "month": None, "timeframe": None}

    # explicit year
    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        res["year"] = int(m.group(1))

    # last year
    if "last year" in q or "previous year" in q:
        res["year"] = now.year - 1

    # last quarter
    if "last quarter" in q or "previous quarter" in q:
        current_q = (now.month - 1) // 3 + 1
        prev_q = current_q - 1 or 4
        prev_y = now.year - 1 if prev_q == 4 else now.year
        res["quarter"] = prev_q
        res["year"] = prev_y

    # explicit quarter
    m = re.search(r'(?:q|quarter)[^\d]*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        res["quarter"] = int(m.group(1))
        if m.group(2):
            res["year"] = int(m.group(2))

    # last month
    if "last month" in q:
        prev = now.replace(day=1) - timedelta(days=1)
        res["month"] = prev.month
        res["year"] = prev.year

    # explicit month name
    months = {m: i for i, m in enumerate(
        ["", "january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"]
    )}
    for name, idx in months.items():
        if name and name in q:
            res["month"] = idx
            if not res["year"]:
                res["year"] = now.year

    return res


# ------------------------------------------------------
# Detect Top N
# ------------------------------------------------------
def detect_top_n(question: str):
    q = question.lower()
    m = re.search(r'top\s+(\d+)', q)
    if m:
        return int(m.group(1))
    return None


# ------------------------------------------------------
# Detect Sorting Intent (highest/lowest/least/biggest)
# ------------------------------------------------------
def detect_sort_intent(question: str):
    q = question.lower()

    DESC_WORDS = [
        "highest", "top", "maximum", "max", "greatest", "largest",
        "most", "bigger", "greater", "biggest", "strongest"
    ]

    ASC_WORDS = [
        "lowest", "least", "minimum", "min", "smallest",
        "bottom", "weaker", "less", "smaller"
    ]

    for w in DESC_WORDS:
        if w in q:
            return "DESC"

    for w in ASC_WORDS:
        if w in q:
            return "ASC"

    return None


# ------------------------------------------------------
# Dimension Filters & Grouping Keywords
# ------------------------------------------------------
DIRECT_MAP = {
    "customer": "CustomerName",
    "branch": "BranchCode",
    "company": "CompanyCode",
    "department": "DeptCode",
    "country": "CountryName",
    "transport": "TransportMode",
    "transport mode": "TransportMode",
    "customer group": "CustomerGroupName",
    "lead group": "CustomerLeadGroupName",
    "product": "ProductLevel1",
    "product level 1": "ProductLevel1",
    "product level 2": "ProductLevel2",
    "product level 3": "ProductLevel3"
}


def detect_dimension_filters(question: str):
    q = question.lower()
    filters = []

    # Transport mode (AIR / SEA / COU / ROA / NOJ / FSA)
    tmodes = ["air", "sea", "roa", "cou", "noj", "fsa"]
    for mode in tmodes:
        if f" {mode} " in q or q.endswith(mode):
            filters.append({
                "column": "TransportMode",
                "operator": "=",
                "value": mode.upper()
            })

    # Customer filter
    m = re.search(r"customer\s+([a-z0-9 _-]+)", q)
    if m:
        filters.append({
            "column": "CustomerName",
            "operator": "=",
            "value": m.group(1).strip().upper()
        })

    return filters


def detect_group_column(question: str):
    q = question.lower()
    for key, col in DIRECT_MAP.items():
        if key in q:
            return col

    # fallback
    m = re.search(r"by\s+([a-z0-9 _-]+)", q)
    if m:
        phrase = m.group(1).replace(" ", "").lower()
        for key, col in DIRECT_MAP.items():
            if key.replace(" ", "") in phrase:
                return col

    return None


# ------------------------------------------------------
# Business Metric Detection
# ------------------------------------------------------
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

    if "profit" in q:
        return "JobProfit"
    if "revenue" in q:
        return "REVAmount"

    return nums[0] if nums else None


# ------------------------------------------------------
# LLM Client
# ------------------------------------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


# ------------------------------------------------------
# MAIN: Convert Question → Query Plan
# ------------------------------------------------------
def extract_query(question: str):
    schema = get_schema()
    nums = numeric_columns()
    cats = categorical_columns()

    time_ctx = parse_time_from_text(question)
    q_lower = question.lower()

    # BUSINESS METRIC PRIORITY
    bm_key = detect_business_metric_key(question)
    bm_meta = BUSINESS_METRICS.get(bm_key) if bm_key else None

    agg = bm_meta["default_agg"] if bm_meta else "sum"

    # SELECT ENTRY
    if bm_meta:
        select_entry = {
            "column": bm_meta["base_column"],
            "expression": bm_meta["expression"],
            "aggregation": agg,
            "alias": bm_meta["alias"]
        }
    else:
        metric_col = detect_metric_column(question)
        select_entry = {
            "column": metric_col,
            "expression": None,
            "aggregation": agg,
            "alias": metric_col
        }

    # GROUPING
    group_col = detect_group_column(question)

    # LIMIT
    top_n = detect_top_n(question)

    plan = {
        "select": [select_entry],
        "filters": [],
        "group_by": [group_col] if group_col else [],
        "order_by": [],
        "limit": top_n
    }

    # TIME FILTERS
    if time_ctx["year"]:
        plan["filters"].append({
            "column": "FinancialYear",
            "operator": "=",
            "value": time_ctx["year"]
        })
    if time_ctx["quarter"]:
        plan["filters"].append({
            "column": "FinancialQuarter",
            "operator": "=",
            "value": time_ctx["quarter"]
        })
    if time_ctx["month"]:
        plan["filters"].append({
            "column": "FinancialMonth",
            "operator": "=",
            "value": time_ctx["month"]
        })

    # DIMENSION FILTERS
    dim_filters = detect_dimension_filters(question)
    if dim_filters:
        plan["filters"].extend(dim_filters)

    # SORTING INTENT (highest/lowest)
    sort_dir = detect_sort_intent(question)
    if sort_dir:
        plan["order_by"] = [{
            "column": select_entry["alias"],
            "direction": sort_dir
        }]

    return plan
