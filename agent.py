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
    return [
        c for c, t in schema.items()
        if t in ('decimal', 'numeric', 'money', 'float', 'int', 'bigint', 'smallint')
    ]


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
# Business metric rules
# --------------------------------
BUSINESS_METRICS = {
    "revenue": {
        "keywords": [
            "revenue", "total revenue", "sales", "turnover", "income", "rev"
        ],
        # full SQL expression
        "expression": "[REVAmount] + [WIPAmount]",
        "base_column": "REVAmount",
        "default_agg": "sum",
        "alias": "total_revenue",
    },
    "cost": {
        "keywords": [
            "cost", "total cost", "expense", "expenses"
        ],
        "expression": "[CSTAmount] + [ACRAmount]",
        "base_column": "CSTAmount",
        "default_agg": "sum",
        "alias": "total_cost",
    },
    "profit": {
        "keywords": [
            "profit", "margin", "jobprofit", "total profit"
        ],
        "expression": "[JobProfit]",
        "base_column": "JobProfit",
        "default_agg": "sum",
        "alias": "total_profit",
    },
}

BUSINESS_RULES_TEXT = """
Key business metrics and rules:

- Revenue = REVAmount + WIPAmount
  * REVAmount: revenue amount of the job.
  * WIPAmount: work-in-progress revenue adjustment.

- Cost = CSTAmount + ACRAmount
  * CSTAmount: cost of service.
  * ACRAmount: additional cost recorded.

- Profit = JobProfit
  * JobProfit: profit of the job.
"""

COLUMN_DESCRIPTIONS_TEXT = """
Important columns:
- JobNumber: identifies shipment / brokerage / warehouse job.
- TransactionMonth / TransactionYear: when the customer transaction happened.
- FinancialMonth / FinancialQuarter / FinancialYear: fiscal calendar for the job.
- CountryName / CountryGroup: customer's country & group.
- CompanyCode, BranchCode, DeptCode: company, branch, department codes.
- TransportMode: SEA, AIR, ROA, COU, FSA, NOJ, NULL.
- ContainerMode: container type (BBK, FCL, LCL, LTL, etc.).
- Direction: import / export / NOJ / others.
- JobType: shipment / warehouse / brokerage.
- JobLevel1/2/3: freight / non-freight and detailed job type.
- ProductLevel1/2/3: product category hierarchy (Air Export, Sea Import, etc.).
- ActualVolume, UnitOfVolume: job volume and unit.
- ActualChargeable, ChargeableUnit: chargeable amount and unit.
- OriginCountry / DestinationCountry / Incoterm / OriginContinent / DestinationContinent.
- ACRAmount: accrual amount.
- CSTAmount: cost of the shipment.
- WIPAmount: work-in-progress revenue.
- REVAmount: revenue amount of job.
- JobProfit: profit of job.
"""


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
# Simple helpers
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
    m = re.search(r'(\d+)\s+(?:customers|clients|branches|rows|records|jobs)', q)
    if m:
        return int(m.group(1))
    return None


def detect_business_metric_key(question: str):
    """Return 'revenue', 'cost', 'profit', ... if question matches."""
    q = question.lower()
    for key, meta in BUSINESS_METRICS.items():
        for kw in meta.get("keywords", []):
            if kw in q:
                return key
    return None


def detect_metric_column(question: str):
    """Fallback numeric column if no business metric is matched."""
    q = question.lower()
    schema = get_schema()
    nums = numeric_columns()

    # direct mention
    for col in nums:
        if col.lower() in q:
            return col

    # simple synonyms
    if 'profit' in q and 'jobprofit' in schema:
        return 'JobProfit'
    if ('revenue' in q or 'sales' in q or 'turnover' in q) and 'REVAmount' in schema:
        return 'REVAmount'

    return nums[0] if nums else None


def detect_group_column(question: str):
    q = question.lower()
    cats = categorical_columns()
    # words after "by ... "
    m = re.search(r'\bby\s+([a-z0-9 _-]{2,40})', q)
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
    if 'country' in q and 'CountryName' in cats:
        return 'CountryName'
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
    Returns a generic query plan:

    {
      "select": [
        {
          "column": "REVAmount",
          "expression": "[REVAmount] + [WIPAmount]",
          "aggregation": "sum",
          "alias": "total_revenue",
          "metric_key": "revenue"
        }
      ],
      "filters": [...],
      "group_by": [...],
      "order_by": [...],
      "limit": 5
    }
    """
    schema = get_schema()
    nums = numeric_columns()
    cats = categorical_columns()
    time_ctx = parse_time_from_text(question)

    q_lower = question.lower()

    # --- 1. Choose metric (business first) ---
    bm_key = detect_business_metric_key(question)
    bm_meta = BUSINESS_METRICS.get(bm_key) if bm_key else None

    # aggregation default
    agg = "sum"
    if any(w in q_lower for w in ["average", "avg", "mean"]):
        agg = "avg"
    if any(w in q_lower for w in ["count", "how many", "number of"]):
        agg = "count"
    if any(w in q_lower for w in ["max", "maximum", "highest", "largest"]):
        agg = "max"
    if any(w in q_lower for w in ["min", "minimum", "lowest", "smallest"]):
        agg = "min"

    select_entry = None

    if bm_meta:
        base_col = bm_meta.get("base_column")
        if base_col and base_col not in nums:
            base_col = None

        select_entry = {
            "column": base_col,
            "expression": bm_meta.get("expression"),
            "aggregation": bm_meta.get("default_agg", agg),
            "alias": bm_meta.get("alias") or (bm_key or "value"),
            "metric_key": bm_key,
        }
    else:
        metric_col = detect_metric_column(question)
        if metric_col:
            select_entry = {
                "column": metric_col,
                "expression": None,
                "aggregation": agg,
                "alias": metric_col,
                "metric_key": None,
            }

    group_col = detect_group_column(question)
    top_n = detect_top_n(question)

    base_plan = {
        "select": [select_entry] if select_entry else [],
        "filters": [],
        "group_by": [group_col] if group_col else [],
        "order_by": [],
        "limit": top_n,
    }

    # --- 2. Ask Groq to improve plan (but do NOT let it break our business metrics) ---
    client = get_client()
    if client and base_plan["select"]:
        try:
            prompt = f"""
You are a senior analytics engineer for a freight forwarding / logistics company.

Convert the user's question into a JSON description of a SQL query
over a single table called {TABLE_NAME}.

Use ONLY these columns:
NUMERIC_COLUMNS = {nums}
CATEGORICAL_COLUMNS = {cats}

Column descriptions:
{COLUMN_DESCRIPTIONS_TEXT}

Business rules:
{BUSINESS_RULES_TEXT}

JSON format (no extra text):
{{
  "select": [
    {{
      "column": "<numeric column or null>",
      "expression": "<SQL expression or null>",
      "aggregation": "sum|avg|max|min|count",
      "alias": "<short_name>"
    }}
  ],
  "filters": [
    {{"column": "<column>", "operator": "=|!=|>|<|>=|<=|in|between|like", "value": "<scalar or list>"}}
  ],
  "group_by": ["<dimension columns>"],
  "order_by": [{{"column": "<column or alias>", "direction": "asc|desc"}}],
  "limit": <integer or null>
}}

Guidelines:
- Use only listed columns.
- For revenue / total revenue / sales: use expression "REVAmount + WIPAmount".
- For cost / total cost: use expression "CSTAmount + ACRAmount".
- For profit / margin: use column "JobProfit".
- If user says "top N", set limit to that N.
- If user asks "by X", add that column to group_by.
- If unsure about filters, leave filters empty.

User question:
\"\"\"{question}\"\"\" 

Return ONLY valid JSON.
"""
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            llm_plan = json.loads(raw)

            if isinstance(llm_plan, dict):
                # we NEVER let LLM override our select for business metrics
                keys_to_merge = ["filters", "group_by", "order_by", "limit"]
                if not bm_meta:
                    keys_to_merge.append("select")
                for k in keys_to_merge:
                    v = llm_plan.get(k)
                    if v not in (None, "", []):
                        base_plan[k] = v
        except Exception as e:
            print("Groq plan generation failed, using fallback:", e)

    plan = base_plan

    # --- 3. Inject time filters (override LLM) ---
    filters = plan.get("filters") or []
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

    # --- 4. Normalise SELECT and re-apply business rules ---
    selects = plan.get("select") or []

    def is_valid_select(sel):
        if not isinstance(sel, dict):
            return False
        col = sel.get("column")
        expr = sel.get("expression")
        if expr:
            return True
        if col and col in nums:
            return True
        return False

    valid_selects = [s for s in selects if is_valid_select(s)]

    if not valid_selects:
        # rebuild with our logic
        bm_key2 = detect_business_metric_key(question)
        bm_meta2 = BUSINESS_METRICS.get(bm_key2) if bm_key2 else None

        agg2 = "sum"
        if any(w in q_lower for w in ["average", "avg", "mean"]):
            agg2 = "avg"
        if any(w in q_lower for w in ["count", "how many", "number of"]):
            agg2 = "count"
        if any(w in q_lower for w in ["max", "maximum", "highest", "largest"]):
            agg2 = "max"
        if any(w in q_lower for w in ["min", "minimum", "lowest", "smallest"]):
            agg2 = "min"

        if bm_meta2:
            base_col = bm_meta2.get("base_column")
            if base_col and base_col not in nums:
                base_col = None
            valid_selects = [{
                "column": base_col,
                "expression": bm_meta2.get("expression"),
                "aggregation": bm_meta2.get("default_agg", agg2),
                "alias": bm_meta2.get("alias") or (bm_key2 or "value"),
                "metric_key": bm_key2,
            }]
        else:
            metric_col = detect_metric_column(question)
            if metric_col:
                valid_selects = [{
                    "column": metric_col,
                    "expression": None,
                    "aggregation": agg2,
                    "alias": metric_col,
                    "metric_key": None,
                }]

    if not valid_selects and nums:
        valid_selects = [{
            "column": nums[0],
            "expression": None,
            "aggregation": "sum",
            "alias": nums[0],
            "metric_key": None,
        }]

    # final alias fix
    for s in valid_selects:
        if not s.get("alias"):
            if s.get("metric_key"):
                s["alias"] = s["metric_key"]
            elif s.get("column"):
                s["alias"] = s["column"]
            else:
                s["alias"] = "value"

    plan["select"] = valid_selects

    # --- 5. Clean ORDER BY and LIMIT ---
    order_by = plan.get("order_by") or []
    cleaned_ob = []
    for ob in order_by:
        if not isinstance(ob, dict):
            continue
        col = ob.get("column")
        if not col:
            continue
        direction = ob.get("direction", "desc").lower()
        if direction not in ("asc", "desc"):
            direction = "desc"
        cleaned_ob.append({"column": col, "direction": direction})
    plan["order_by"] = cleaned_ob

    limit = plan.get("limit")
    if isinstance(limit, str) and limit.isdigit():
        limit = int(limit)
    if not isinstance(limit, int):
        limit = detect_top_n(question)
    plan["limit"] = limit

    return plan
