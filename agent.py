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
    """Read INFORMATION_SCHEMA for the target table."""
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
    """Treat everything that is not numeric/date as a dimension."""
    schema = get_schema()
    nums = set(numeric_columns())
    cats = [
        c for c, t in schema.items()
        if c not in nums and t not in ('date', 'datetime', 'smalldatetime', 'datetime2')
    ]
    return cats


# --------------------------------
# Business metrics (Option A with SQL expression)
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
        "keywords": ["profit", "margin", "jobprofit"],
        "expression": "[JobProfit]",
        "base_column": "JobProfit",
        "default_agg": "sum",
        "alias": "total_profit"
    },
}

BUSINESS_RULES_TEXT = """
Business metric rules:
- Revenue = REVAmount + WIPAmount
- Cost = CSTAmount + ACRAmount
- Profit = JobProfit

Time:
- FinancialMonth runs from Apr(1) to Mar(12)
"""

COLUMN_DESCRIPTIONS_TEXT = """
TransportMode = AIR, SEA, COU, ROA, NOJ, FSA
ProductLevel1: Air Export, Sea Import, etc.
CustomerName, CountryName, BranchCode etc.
All shipment financial metrics available.
"""


# --------------------------------
# Time parsing helper
# --------------------------------
def parse_time_from_text(question: str):
    q = question.lower()
    now = datetime.utcnow()
    res = {"year": None, "quarter": None, "month": None, "timeframe": None}

    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        res["year"] = int(m.group(1))

    if 'previous year' in q or 'last year' in q:
        res["year"] = now.year - 1
        res["timeframe"] = "previous_year"

    if 'last quarter' in q or 'previous quarter' in q:
        current_q = (now.month - 1) // 3 + 1
        prev_q = current_q - 1
        prev_year = now.year
        if prev_q == 0:
            prev_q = 4
            prev_year -= 1
        res["quarter"] = prev_q
        res["year"] = prev_year
        res["timeframe"] = "last_quarter"

    m = re.search(r'(?:q|quarter)[^\d]*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        res["quarter"] = int(m.group(1))
        if m.group(2):
            res["year"] = int(m.group(2))

    if 'last month' in q or 'previous month' in q:
        prev = now.replace(day=1) - timedelta(days=1)
        res["month"] = prev.month
        res["year"] = prev.year
        res["timeframe"] = "last_month"

    months = {
        name.lower(): idx
        for idx, name in enumerate(
            ["", "january","february","march","april","may","june",
             "july","august","september","october","november","december"]
        )
    }
    for name, idx in months.items():
        if name and name in q:
            res["month"] = idx
            if not res["year"]:
                res["year"] = now.year
            break

    return res


# --------------------------------
# Detect top N
# --------------------------------
def detect_top_n(question: str):
    q = question.lower()
    m = re.search(r'top\s+(\d+)', q)
    if m:
        return int(m.group(1))
    return None


# --------------------------------
# Detect business metric key
# --------------------------------
def detect_business_metric_key(question: str):
    q = question.lower()
    for key, meta in BUSINESS_METRICS.items():
        for kw in meta.get("keywords", []):
            if kw in q:
                return key
    return None


# --------------------------------
# Detect fallback numeric metric
# --------------------------------
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


# --------------------------------
# Detect group column
# --------------------------------
def detect_group_column(question: str):
    q = question.lower()
    cats = categorical_columns()

    m = re.search(r'\bby\s+([a-z0-9 _-]{1,40})', q)
    if m:
        candidate = m.group(1).strip()
        cand2 = candidate.replace(" ", "").lower()
        for col in cats:
            if cand2 in col.lower():
                return col

    if 'transport' in q and 'TransportMode' in cats:
        return 'TransportMode'

    if 'customer' in q and 'CustomerName' in cats:
        return 'CustomerName'

    if 'country' in q and 'CountryName' in cats:
        return 'CountryName'

    return None


# --------------------------------
# Detect category VALUE (AIR / SEA / INDIA / XYZ)
# --------------------------------
def detect_category_value(question: str, group_col: str):
    if not group_col:
        return None

    q = question.lower()

    # direct: "transport mode air"
    m = re.search(rf"{group_col.lower()}[^\w]+([a-z0-9 _-]+)", q)
    if m:
        return m.group(1).strip().upper()

    # general: "for air", "for india"
    m = re.search(r"for\s+([a-z0-9 _-]+)", q)
    if m:
        value = m.group(1).strip().upper()
        if "YEAR" not in value and "MONTH" not in value:
            return value

    # special list for transport codes
    transport_codes = ["AIR", "SEA", "COU", "ROA", "NOJ", "FSA"]
    for code in transport_codes:
        if code.lower() in q:
            return code

    return None


# --------------------------------
# Groq client
# --------------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# --------------------------------
# Main: question → plan
# --------------------------------
def extract_query(question: str):
    schema = get_schema()
    nums = numeric_columns()
    cats = categorical_columns()
    time_ctx = parse_time_from_text(question)
    q_lower = question.lower()

    bm_key = detect_business_metric_key(question)
    bm_meta = BUSINESS_METRICS.get(bm_key) if bm_key else None

    agg = "sum"
    if "avg" in q_lower or "average" in q_lower:
        agg = "avg"

    if "count" in q_lower:
        agg = "count"

    # build select
    if bm_meta:
        select_entry = {
            "column": bm_meta["base_column"],
            "expression": bm_meta["expression"],
            "aggregation": bm_meta["default_agg"],
            "alias": bm_meta["alias"],
            "metric_key": bm_key
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
    category_value = detect_category_value(question, group_col)
    top_n = detect_top_n(question)

    plan = {
        "select": [select_entry],
        "filters": [],
        "group_by": [],
        "category_value": category_value,
        "order_by": [],
        "limit": top_n
    }

    # If user specifies a value → DO NOT group
    if group_col and not category_value:
        plan["group_by"] = [group_col]

    # If value exists → add filter
    if group_col and category_value:
        plan["filters"].append({
            "column": group_col,
            "operator": "=",
            "value": category_value
        })

    # Inject time filters
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

    return plan
