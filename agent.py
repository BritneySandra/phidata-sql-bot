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

# Your confirmed categorical columns
CATEGORICAL_COLUMNS = [
    "TransportMode", "ContainerMode", "Direction",
    "ProductLevel1", "ProductLevel2", "ProductLevel3",
    "BranchCode", "AdjustedBranchCode",
    "DeptCode",
    "CustomerName", "CustomerGroup", "CustomerGroupName",
    "CompanyCode",
    "CountryName", "DestinationCountry", "OriginCountry",
    "JobType",
    "JobLevel1", "JobLevel2", "JobLevel3",
    "Currency",
    "Incoterm",
    "BusinessType",
    "ConsigneeImporterFullName",
    "ConsignorShipperSuplierFullName"
]


# -------------------------
# Load schema (column -> data_type)
# -------------------------
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
        # fallback: build from known categorical columns
        return {c: "varchar" for c in CATEGORICAL_COLUMNS}


_SCHEMA = {}


def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA


# -------------------------
# Numeric columns detection
# -------------------------
def numeric_columns():
    schema = get_schema()
    nums = [c for c, t in schema.items()
            if t in ('decimal', 'numeric', 'money', 'float', 'int', 'bigint', 'smallint')]
    return nums


def categorical_columns():
    schema = get_schema()
    cats = []
    for col, t in schema.items():
        if t in ('varchar', 'nvarchar', 'char', 'text'):
            cats.append(col)
    # ensure your explicit ones are included
    for col in CATEGORICAL_COLUMNS:
        if col in schema and col not in cats:
            cats.append(col)
    return cats


# -------------------------
# Time parsing helper (for 'last year', 'last quarter', etc.)
# -------------------------
def parse_time_from_text(question):
    q = question.lower()
    now = datetime.utcnow()
    res = {"year": None, "quarter": None, "month": None, "timeframe": None}

    # explicit year like 2023, 2024
    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        res["year"] = int(m.group(1))

    # last year / previous year
    if 'previous year' in q or 'last year' in q:
        res["year"] = now.year - 1
        res["timeframe"] = "previous_year"

    # last quarter / previous quarter
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

    # explicit quarter like Q1 2024
    m = re.search(r'(?:q|quarter)[^\d]*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        res["quarter"] = int(m.group(1))
        if m.group(2):
            res["year"] = int(m.group(2))

    # last month
    if 'last month' in q or 'previous month' in q:
        prev = now.replace(day=1) - timedelta(days=1)
        res["month"] = prev.month
        res["year"] = prev.year
        res["timeframe"] = "last_month"

    # month name like "march", "october"
    months = {m.lower(): i for i, m in enumerate([
        "", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ])}
    for name, idx in months.items():
        if name and name in q:
            res["month"] = idx
            if not res["year"]:
                res["year"] = now.year
            break

    return res


# -------------------------
# Fallback metric detection (if LLM fails)
# -------------------------
def find_metric_from_text(question):
    q = question.lower()
    schema = get_schema()
    nums = numeric_columns()

    # try explicit column name
    for col in nums:
        if col.lower() in q:
            return col

    # simple keyword heuristics
    if 'profit' in q and 'JobProfit' in schema:
        return 'JobProfit'
    if 'revenue' in q and 'REVAmount' in schema:
        return 'REVAmount'

    return nums[0] if nums else None


# -------------------------
# Groq client safe init
# -------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -------------------------
# Main: build a JSON query plan (Option B)
# -------------------------
def extract_query(question: str) -> dict:
    """
    Returns a generic query plan:

    {
      "select": [
        {"column": "JobProfit", "aggregation": "sum"}
      ],
      "filters": [
        {"column": "FinancialYear", "operator": "=", "value": 2024}
      ],
      "group_by": ["TransportMode"],
      "order_by": [{"column": "JobProfit", "direction": "desc"}],
      "limit": 50
    }
    """
    schema = get_schema()
    num_cols = numeric_columns()
    cat_cols = categorical_columns()
    client = get_client()

    plan = {}

    # -------- Try LLM first --------
    if client:
        cols_desc = [
            f"- {name}: {dtype}"
            for name, dtype in schema.items()
        ]
        numeric_list = ", ".join(num_cols) or "(none)"
        categorical_list = ", ".join(cat_cols) or "(none)"

        prompt = f"""
You are an assistant that converts a business question into a JSON query plan
for a SQL Server table named {TABLE_NAME}.

Table columns and types:
{chr(10).join(cols_desc)}

Numeric columns (valid for aggregation): {numeric_list}
Categorical columns (valid for grouping/filtering): {categorical_list}

Return ONLY valid JSON with this exact structure:

{{
  "select": [
    {{"column": "<one column name>", "aggregation": "sum|avg|max|min|count|none"}}
  ],
  "filters": [
    {{
      "column": "<column name>",
      "operator": "=|!=|>|<|>=|<=|in|between|like",
      "value": "<single value, list of values for 'in', or [from,to] for 'between'>"
    }}
  ],
  "group_by": ["<col1>", "<col2>"],
  "order_by": [
    {{"column": "<column name>", "direction": "asc|desc"}}
  ],
  "limit": <integer or null>
}}

Rules:
- Use only column names from the schema provided.
- Use only numeric columns in "select" when using aggregations (sum/avg/max/min/count).
- "aggregation": "none" means select the raw column without aggregation.
- Use "filters" for any conditions (year, quarter, month, specific categories, etc.).
- For time filters:
    - Use FinancialYear, FinancialQuarter, FinancialMonth if the user mentions year/quarter/month.
- If the user says "by X" or "breakdown by X", put X in "group_by".
- If the user asks for "top N" or "highest", use "order_by" + "limit".
- If the question compares categories (e.g. SEA vs AIR), use an "in" filter on the correct categorical column.
- If something is not mentioned, leave the corresponding array empty or null.

User Question: "{question}"

Return ONLY the JSON. Do NOT include explanations or markdown.
"""
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            plan = json.loads(raw)
        except Exception as e:
            print("LLM parse failed, using fallback heuristics:", e)
            plan = {}

    # -------- Fallback / Validation --------
    if not isinstance(plan, dict):
        plan = {}

    select = plan.get("select") or []
    filters = plan.get("filters") or []
    group_by = plan.get("group_by") or []
    order_by = plan.get("order_by") or []
    limit = plan.get("limit")

    # Ensure we have at least one metric in select
    if not select:
        metric = find_metric_from_text(question)
        if metric:
            select = [{"column": metric, "aggregation": "sum"}]
        elif num_cols:
            select = [{"column": num_cols[0], "aggregation": "count"}]

    valid_cols = set(schema.keys())

    # Clean select
    clean_select = []
    for s in select:
        col = s.get("column")
        if col not in valid_cols:
            continue
        agg = (s.get("aggregation") or "sum").lower()
        if agg not in ("sum", "avg", "max", "min", "count", "none"):
            agg = "sum"
        clean_select.append({"column": col, "aggregation": agg})
    select = clean_select

    # Clean filters
    clean_filters = []
    for f in filters:
        col = f.get("column")
        if col not in valid_cols:
            continue
        op = (f.get("operator") or "=").lower()
        val = f.get("value")
        clean_filters.append({
            "column": col,
            "operator": op,
            "value": val
        })
    filters = clean_filters

    # Inject time filters from explicit NLP if LLM missed them
    tinfo = parse_time_from_text(question)
    if tinfo.get("year") and "FinancialYear" in valid_cols and not any(f["column"] == "FinancialYear" for f in filters):
        filters.append({"column": "FinancialYear", "operator": "=", "value": tinfo["year"]})
    if tinfo.get("quarter") and "FinancialQuarter" in valid_cols and not any(f["column"] == "FinancialQuarter" for f in filters):
        filters.append({"column": "FinancialQuarter", "operator": "=", "value": tinfo["quarter"]})
    if tinfo.get("month") and "FinancialMonth" in valid_cols and not any(f["column"] == "FinancialMonth" for f in filters):
        filters.append({"column": "FinancialMonth", "operator": "=", "value": tinfo["month"]})

    # Clean group_by
    if isinstance(group_by, str):
        group_by = [group_by]
    group_by = [g for g in (group_by or []) if g in valid_cols]

    # Clean order_by
    clean_order = []
    if isinstance(order_by, dict):
        order_by = [order_by]
    for o in (order_by or []):
        col = o.get("column")
        if col not in valid_cols:
            continue
        direction = (o.get("direction") or "asc").lower()
        clean_order.append({"column": col, "direction": direction})
    order_by = clean_order

    # Clean limit
    if isinstance(limit, str) and limit.isdigit():
        limit = int(limit)
    if not isinstance(limit, int):
        limit = None

    return {
        "select": select,
        "filters": filters,
        "group_by": group_by,
        "order_by": order_by,
        "limit": limit,
    }
