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
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{TABLE_NAME}'
        """)
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
        "keywords": ["cost", "total cost", "expense"],
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
    }
}


# --------------------------------
# Time parsing helper
# --------------------------------
def parse_time_from_text(question: str):
    q = question.lower()
    now = datetime.utcnow()

    result = {"year": None, "quarter": None, "month": None}

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        result["year"] = int(m.group(1))

    if "last year" in q or "previous year" in q:
        result["year"] = now.year - 1

    if "last quarter" in q or "previous quarter" in q:
        current_q = (now.month - 1) // 3 + 1
        prev_q = current_q - 1 or 4
        prev_y = now.year - 1 if prev_q == 4 else now.year
        result["quarter"] = prev_q
        result["year"] = prev_y

    m = re.search(r"(?:q|quarter)[^\d]*([1-4])(?:[^0-9]+(20\d{2}))?", q)
    if m:
        result["quarter"] = int(m.group(1))
        if m.group(2):
            result["year"] = int(m.group(2))

    if "last month" in q:
        prev = now.replace(day=1) - timedelta(days=1)
        result["month"] = prev.month
        result["year"] = prev.year

    months = {
        name: idx for idx, name in enumerate([
            "", "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ])
    }
    for name, idx in months.items():
        if name and name in q:
            result["month"] = idx
            if not result["year"]:
                result["year"] = now.year

    return result


# --------------------------------
# Top N detection
# --------------------------------
def detect_top_n(question: str):
    m = re.search(r"top\s+(\d+)", question.lower())
    return int(m.group(1)) if m else None


# --------------------------------
# Dimension filters
# --------------------------------
def detect_dimension_filters(question: str):
    q = question.lower()
    filters = []

    transport_modes = ["air", "sea", "roa", "cou", "noj", "fsa"]

    for mode in transport_modes:
        if f" {mode} " in q or q.endswith(mode):
            filters.append({"column": "TransportMode", "operator": "=", "value": mode.upper()})

    return filters


# --------------------------------
# BUSINESS METRIC detection
# --------------------------------
def detect_business_metric_key(question: str):
    q = question.lower()
    for key, meta in BUSINESS_METRICS.items():
        for word in meta["keywords"]:
            if word in q:
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


# --------------------------------
# UPDATED GROUPING LOGIC (Fix)
# --------------------------------
def detect_group_column(question: str):
    q = question.lower()
    cats = categorical_columns()

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

    for key, col in DIRECT_MAP.items():
        if key in q and col in cats:
            return col

    m = re.search(r"\b(by|each|per)\s+([a-z0-9 _-]+)", q)
    if m:
        target = m.group(2).strip().replace(" ", "").lower()
        for col in cats:
            if target in col.lower().replace(" ", ""):
                return col

    return None


# --------------------------------
# Groq client
# --------------------------------
def get_client():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


# --------------------------------
# MAIN: Convert question → Plan
# --------------------------------
def extract_query(question: str):
    schema = get_schema()
    nums = numeric_columns()
    cats = categorical_columns()

    time_ctx = parse_time_from_text(question)

    metric_key = detect_business_metric_key(question)
    metric_meta = BUSINESS_METRICS.get(metric_key) if metric_key else None

    agg = metric_meta["default_agg"] if metric_meta else "sum"

    if metric_meta:
        select_entry = {
            "column": metric_meta["base_column"],
            "expression": metric_meta["expression"],
            "aggregation": agg,
            "alias": metric_meta["alias"]
        }
    else:
        col = detect_metric_column(question)
        select_entry = {
            "column": col,
            "expression": None,
            "aggregation": agg,
            "alias": col
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

    if time_ctx["year"]:
        plan["filters"].append({"column": "FinancialYear", "operator": "=", "value": time_ctx["year"]})
    if time_ctx["quarter"]:
        plan["filters"].append({"column": "FinancialQuarter", "operator": "=", "value": time_ctx["quarter"]})
    if time_ctx["month"]:
        plan["filters"].append({"column": "FinancialMonth", "operator": "=", "value": time_ctx["month"]})

    dim_filters = detect_dimension_filters(question)
    if dim_filters:
        plan["filters"].extend(dim_filters)

    return plan
