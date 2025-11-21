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
    return [c for c, t in schema.items() if t in ('decimal', 'numeric', 'money', 'float', 'int', 'bigint', 'smallint')]

def categorical_columns():
    """Treat everything that is not numeric/date as a dimension."""
    schema = get_schema()
    nums = set(numeric_columns())
    cats = [c for c, t in schema.items()
            if c not in nums and t not in ('date', 'datetime', 'smalldatetime', 'datetime2')]
    return cats

# --------------------------------
# Time parsing helper
# --------------------------------
def parse_time_from_text(question: str):
    """
    Returns: {"year": int|None, "quarter": int|None, "month": int|None, "timeframe": str|None}
    Resolves 'last year', 'last quarter', month names, etc. to explicit numbers.
    """
    q = question.lower()
    now = datetime.utcnow()
    res = {"year": None, "quarter": None, "month": None, "timeframe": None}

    # explicit year like 2023, 2024
    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        res["year"] = int(m.group(1))

    # last / previous year
    if 'previous year' in q or 'last year' in q:
        res["year"] = now.year - 1
        res["timeframe"] = "previous_year"

    # last / previous quarter
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

    # explicit quarter: Q1 2024, quarter 3 2023, etc.
    m = re.search(r'(?:q|quarter)[^\d]*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        res["quarter"] = int(m.group(1))
        if m.group(2):
            res["year"] = int(m.group(2))

    # last / previous month
    if 'last month' in q or 'previous month' in q:
        prev = now.replace(day=1) - timedelta(days=1)
        res["month"] = prev.month
        res["year"] = prev.year
        res["timeframe"] = "last_month"

    # month name (march, aug 2024 etc.)
    months = {
        name.lower(): idx
        for idx, name in enumerate(
            ["", "january", "february", "march", "april", "may", "june",
             "july", "august", "september", "october", "november", "december"]
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
# Simple helpers for fallback
# --------------------------------
def detect_top_n(question: str):
    """Find 'top 5', 'top 10', 'first 3' etc."""
    q = question.lower()
    m = re.search(r'top\s+(\d+)', q)
    if m:
        return int(m.group(1))
    m = re.search(r'first\s+(\d+)', q)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s+(?:customers|clients|branches|rows|records)', q)
    if m:
        return int(m.group(1))
    return None

def detect_metric(question: str):
    q = question.lower()
    schema = get_schema()
    nums = numeric_columns()

    # direct mention
    for col in nums:
        if col.lower() in q:
            return col

    # simple synonyms
    if 'profit' in q and 'JobProfit' in schema:
        return 'JobProfit'
    if 'revenue' in q or 'sales' in q or 'turnover' in q:
        if 'REVAmount' in schema:
            return 'REVAmount'

    return nums[0] if nums else None

def detect_group_column(question: str):
    q = question.lower()
    cats = categorical_columns()
    # words after "by ..."
    m = re.search(r'\bby\s+([a-z0-9 _-]{2,40})', q)
    candidate = None
    if m:
        candidate = m.group(1).strip()
        candidate_nospace = candidate.replace(" ", "").lower()
        for col in cats:
            if candidate_nospace in col.lower():
                return col
    # simple keywords
    if 'transport' in q and 'TransportMode' in cats:
        return 'TransportMode'
    if 'product' in q and 'ProductLevel1' in cats:
        return 'ProductLevel1'
    if 'branch' in q and 'BranchCode' in cats:
        return 'BranchCode'
    if 'customer' in q and 'CustomerName' in cats:
        return 'CustomerName'
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
# Main: question -> generic JSON plan
# --------------------------------
def extract_query(question: str):
    """
    Returns a *generic* query plan (not tied to cases):

    {
      "select": [
        {"column": "JobProfit", "aggregation": "sum", "alias": "value"}
      ],
      "filters": [
        {"column": "FinancialYear", "operator": "=", "value": 2024},
        {"column": "TransportMode", "operator": "in", "value": ["SEA","AIR"]}
      ],
      "group_by": ["TransportMode"],
      "order_by": [{"column": "value", "direction": "desc"}],
      "limit": 5
    }
    """
    schema = get_schema()
    nums = numeric_columns()
    cats = categorical_columns()
    time_ctx = parse_time_from_text(question)

    # --------------------
    # 1. Default / fallback plan
    # --------------------
    metric = detect_metric(question)
    if not metric:
        metric = nums[0] if nums else None

    agg = "sum"
    q_lower = question.lower()
    if any(w in q_lower for w in ["average", "avg", "mean"]):
        agg = "avg"
    if any(w in q_lower for w in ["count", "how many", "number of"]):
        agg = "count"
    if any(w in q_lower for w in ["max", "maximum", "highest", "largest"]):
        agg = "max"
    if any(w in q_lower for w in ["min", "minimum", "lowest", "smallest"]):
        agg = "min"

    group_col = detect_group_column(question)
    top_n = detect_top_n(question)

    base_plan = {
        "select": [{"column": metric, "aggregation": agg, "alias": "value"}] if metric else [],
        "filters": [],
        "group_by": [group_col] if group_col else [],
        "order_by": [],
        "limit": top_n
    }

    # --------------------
    # 2. Try Groq to refine the plan
    # --------------------
    client = get_client()
    if client and metric:
        try:
            prompt = f"""
You are a senior analytics engineer.

Convert the user's question into a JSON description of a SQL query.

USE ONLY these columns:
NUMERIC_COLUMNS = {nums}
CATEGORICAL_COLUMNS = {cats}

JSON format (no extra text):
{{
  "select": [
    {{"column": "<numeric column>", "aggregation": "sum|avg|max|min|count", "alias": "value"}}
  ],
  "filters": [
    {{"column": "<column>", "operator": "=|!=|>|<|>=|<=|in|between|like", "value": "<scalar or list>"}}
  ],
  "group_by": ["<dimension columns>"],
  "order_by": [{{"column": "<column or alias>", "direction": "asc|desc"}}],
  "limit": <integer or null>
}}

Guidelines:
- ALWAYS use only columns from the lists above.
- If the user says "top 5" or similar, set "limit" to that integer.
- If user asks "by transport mode / by customer / by branch", add that column to group_by.
- For a single overall total, leave group_by empty.
- If unsure about filters, leave "filters" empty (they will be added later).

User question:
\"\"\"{question}\"\"\"

Return ONLY valid JSON.
"""
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256
            )
            raw = resp.choices[0].message.content.strip()
            llm_plan = json.loads(raw)

            # Merge LLM plan over fallback
            if isinstance(llm_plan, dict):
                base_plan.update({k: v for k, v in llm_plan.items() if v not in (None, "", [])})
        except Exception as e:
            print("Groq plan generation failed, using fallback:", e)

    plan = base_plan

    # --------------------
    # 3. Inject time filters (our logic overrides any LLM mistakes)
    # --------------------
    filters = plan.get("filters") or []
    # Drop any time filters the LLM might have tried to guess
    filters = [
        f for f in filters
        if f.get("column") not in ("FinancialYear", "FinancialQuarter", "FinancialMonth")
    ]

    if time_ctx.get("year"):
        filters.append({"column": "FinancialYear", "operator": "=", "value": time_ctx["year"]})
    if time_ctx.get("quarter"):
        filters.append({"column": "FinancialQuarter", "operator": "=", "value": time_ctx["quarter"]})
    if time_ctx.get("month"):
        filters.append({"column": "FinancialMonth", "operator": "=", "value": time_ctx["month"]})

    plan["filters"] = filters

    # Ensure select exists
    if not plan.get("select"):
        if metric:
            plan["select"] = [{"column": metric, "aggregation": agg, "alias": "value"}]

    # Ensure limit is int or None
    limit = plan.get("limit")
    if isinstance(limit, str) and limit.isdigit():
        plan["limit"] = int(limit)
    if not isinstance(plan.get("limit"), int):
        plan["limit"] = None

    return plan
