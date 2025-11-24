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

# ================================================================
# 1. SCHEMA LOADING
# ================================================================
def load_sql_schema():
    """Read INFORMATION_SCHEMA for the target table and return {column: data_type}."""
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
    """
    Treat everything that is not numeric/date as a dimension (category).
    """
    schema = get_schema()
    nums = set(numeric_columns())
    cats = [
        c for c, t in schema.items()
        if c not in nums and t not in ('date', 'datetime', 'smalldatetime', 'datetime2', 'datetimeoffset')
    ]
    return cats


# ================================================================
# 2. COLUMN DESCRIPTIONS (for LLM understanding only)
# ================================================================
COLUMN_DESCRIPTIONS = {
    "JobPK": "Unique identifier for each job (primary key).",
    "JobNumber": "Identifier for brokerage, shipment and warehouse jobs.",
    "TransactionMonth": "Month when the customer transaction happened (1-12).",
    "TransactionYear": "Year when the customer transaction happened.",
    "FinancialMonth": "Financial month (1=Apr, 2=May, ..., 12=Mar).",
    "FinancialQuarter": "Financial quarter number.",
    "FinancialYear": "Financial year.",
    "FYMonth": "Combined FY year and month, e.g. FY-18-Dec.",
    "FYQuarter": "Combined FY year and quarter, e.g. FY-18-Q4.",
    "FYYear": "Financial year label, e.g. FY-25.",
    "CountryPK": "Unique identifier for country (primary key).",
    "CountryName": "Customer's country.",
    "CountryGroup": "Customer country group.",
    "CompanyPK": "Unique identifier for company (primary key).",
    "CompanyCode": "Customer's company code.",
    "BranchPK": "Unique identifier for branch (primary key).",
    "BranchCode": "Customer's branch code.",
    "DeptPK": "Unique identifier for department (primary key).",
    "DeptCode": "Customer's department code.",
    "JobStatus": "Status of the shipment/job.",
    "JobRevenueStatus": "Revenue/payment status of shipment/job.",
    "TransportMode": "Transport mode of the shipment/job (NULL, AIR, COU, FSA, NOJ, ROA, SEA).",
    "ContainerMode": "Container mode (BBK, BCN, CNT, CON, FCL, FTL, JHB, LCL, LSE, LTL, OBC, ROR, SCN, UNA, NOJ, NULL).",
    "Direction": "Shipment direction: import/export/NOJ/other.",
    "JobType": "Type of job: shipment/warehouse/brokerage.",
    "JobLevel1": "Freight vs non-freight classification.",
    "JobLevel2": "Job level 2 (CrossTrade, Destination, Freight, Origin, Others, PassThrough, Transportation, Warehouse).",
    "JobLevel3": "Job level 3 (CrossTrade, Customs-Export, Customs-Import, Destination Job, Freight-P2P, Origin Job, Others, PassThrough, Transportation, Transportation-Export, Transportation-Import, Warehouse).",
    "ProductLevel1": "High-level product category (Air Export, Air Import, Cartage-Delivery, Cartage-FCL, Cartage-Others, Cartage-Pickup, Cartage-SeaExport, Cartage-SeaImport, Customs, Others, Sea Export, Sea Import).",
    "ProductLevel2": "Second level product category (Cartage-Delivery, Cartage-FCL, Cartage-Others, Cartage-Pickup, Cartage-SeaExport, Cartage-SeaImport, FCL, LCL, LSE).",
    "ProductLevel3": "Third level product category (BBK, BCN, CNT, CON, FCL, FTL, JHB, LCL, LSE, LTL, OBC, ROR, SCN, UNA, NULL).",
    "OriginETD": "Estimated time of departure from origin country.",
    "DestinationETA": "Estimated time of arrival to destination country.",
    "ActualVolume": "Total shipment/job volume.",
    "UnitOfVolume": "Unit of volume (tons, kg, m3, etc.).",
    "ActualChargeable": "Actual chargeable amount of job.",
    "ChargeableUnit": "Unit for chargeable quantity (KG, M3, etc.).",
    "TEU": "TEU volume for the job.",
    "AirVolumeInTons": "Volume in tons for transport mode AIR.",
    "AirVolumeInCBM": "Volume in CBM for AIR.",
    "SeaVolumeLCL": "Volume in LCL for SEA.",
    "SeaVolumeFCL": "Volume in FCL for SEA.",
    "SeaVolumeFCLTEU": "FCL volume in TEU for SEA.",
    "FinalVolumeinCBM": "Final volume of job in cubic meters.",
    "CustomerCode": "Customer code.",
    "CustomerName": "Customer name.",
    "CustomerGroup": "Customer group code.",
    "CustomerGroupName": "Customer group name.",
    "CustomerLeadGroup": "Customer lead group code.",
    "CustomerLeadGroupName": "Customer lead group name.",
    "ConsolNo": "Consolidated job number.",
    "ConsolATA": "Actual time of arrival for consolidated shipment.",
    "ConsolATD": "Actual time of departure for consolidated shipment.",
    "OriginCountry": "Origin country of the shipment.",
    "DestinationCountry": "Destination country of the shipment.",
    "Incoterm": "International commercial terms (trade agreement between buyer and seller).",
    "OriginContinent": "Origin continent.",
    "DestinationContinent": "Destination continent.",
    "ACRAmount": "Accrual amount (additional cost recorded).",
    "CSTAmount": "Cost amount (cost of service).",
    "WIPAmount": "Work-in-progress revenue adjustment.",
    "REVAmount": "Base revenue amount for the job.",
    "JobProfit": "Profit for the job (revenue - cost)."
}

# ================================================================
# 3. BUSINESS METRICS (virtual measures)
# ================================================================
# These are NOT physical columns; they are SQL expressions or aggregate formulas.
BUSINESS_METRICS = {
    "Revenue": {
        "type": "expression",   # row-level expression wrapped in aggregation
        "sql": "(REVAmount + WIPAmount)",
        "description": "Revenue = REVAmount + WIPAmount (base revenue plus WIP adjustment).",
        "default_agg": "sum",
        "synonyms": ["revenue", "total revenue", "sales", "turnover", "income"]
    },
    "Cost": {
        "type": "expression",
        "sql": "(CSTAmount + ACRAmount)",
        "description": "Cost = CSTAmount + ACRAmount (service cost plus accruals).",
        "default_agg": "sum",
        "synonyms": ["cost", "total cost", "expenses", "service cost"]
    },
    "Profit": {
        "type": "column",
        "sql": "JobProfit",
        "description": "Profit = JobProfit.",
        "default_agg": "sum",
        "synonyms": ["profit", "job profit", "margin", "earnings"]
    },
    "ProfitPercent": {
        "type": "aggregate_expression",
        "sql": "SUM(JobProfit) / NULLIF(SUM(CSTAmount + ACRAmount), 0)",
        "description": "Profit % = Total Profit / Total Cost.",
        "default_agg": None,
        "synonyms": ["profit %", "margin %", "profit percentage", "margin percentage", "profit ratio"]
    },
    "TotalJobCount": {
        "type": "aggregate_expression",
        "sql": "COUNT(DISTINCT JobNumber)",
        "description": "Total Job Count = distinct count of JobNumber.",
        "default_agg": None,
        "synonyms": ["total job count", "job count", "number of jobs", "jobs"]
    },
    "TotalVolume": {
        "type": "aggregate_expression",
        # Approximation of your DAX SUMX(VALUES(JobNumber), AVERAGE(ActualVolume)):
        "sql": "SUM(ActualVolume)",
        "description": "Total Volume = sum of ActualVolume (approximation of per-job average volume).",
        "default_agg": None,
        "synonyms": ["total volume", "volume", "shipment volume"]
    },
    "TradeLaneContinent": {
        "type": "expression",
        "sql": "(OriginContinent + '-' + DestinationContinent)",
        "description": "Trade Lane Continent = OriginContinent + '-' + DestinationContinent.",
        "default_agg": None,
        "synonyms": ["trade lane continent", "continent lane"]
    },
    "TradeLaneCountry": {
        "type": "expression",
        "sql": "(OriginCountry + '-' + DestinationCountry)",
        "description": "Trade Lane Country = OriginCountry + '-' + DestinationCountry.",
        "default_agg": None,
        "synonyms": ["trade lane", "trade lane country", "country lane", "origin-destination"]
    },
    "Arrival_Delay_Days": {
        "type": "expression",
        "sql": "DATEDIFF(day, DestinationETA, ConsolATA)",
        "description": "Arrival_Delay_Days = DATEDIFF(day, DestinationETA, ConsolATA).",
        "default_agg": "avg",
        "synonyms": ["arrival delay", "arrival delay days", "eta delay"]
    },
    "Departure_Delay_Days": {
        "type": "expression",
        "sql": "DATEDIFF(day, OriginETD, ConsolATD)",
        "description": "Departure_Delay_Days = DATEDIFF(day, OriginETD, ConsolATD).",
        "default_agg": "avg",
        "synonyms": ["departure delay", "departure delay days", "etd delay"]
    }
}

def get_business_metrics():
    """Expose definitions so sql_builder.py can import and use them."""
    return BUSINESS_METRICS


# ================================================================
# 4. TIME PARSING
# ================================================================
def parse_time_from_text(question: str):
    """
    Returns: {"year": int|None, "quarter": int|None, "month": int|None, "timeframe": str|None}
    Handles 'last year', 'last quarter', month names etc.
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


# ================================================================
# 5. SIMPLE FALLBACK HELPERS
# ================================================================
def detect_top_n(question: str):
    """Find 'top 5', 'top 10', 'first 3', '10 customers', etc."""
    q = question.lower()
    m = re.search(r'top\s+(\d+)', q)
    if m:
        return int(m.group(1))
    m = re.search(r'first\s+(\d+)', q)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s+(?:customers|clients|branches|rows|records|jobs|lanes)', q)
    if m:
        return int(m.group(1))
    return None


def detect_metric(question: str):
    """
    Decide which metric the user is asking for.
    We first check business metrics, then physical numeric columns.
    """
    q = question.lower()
    schema = get_schema()
    nums = numeric_columns()

    # 1) Business metric synonyms (Revenue, Cost, Profit%, etc.)
    for name, meta in BUSINESS_METRICS.items():
        for syn in meta.get("synonyms", []):
            if syn in q:
                return name   # return virtual metric name

    # 2) explicit physical numeric column
    for col in nums:
        if col.lower() in q:
            return col

    # 3) generic words for revenue/profit if not caught above
    if 'profit' in q and 'JobProfit' in schema:
        return 'Profit'
    if ('revenue' in q or 'sales' in q or 'turnover' in q) and 'REVAmount' in schema:
        return 'Revenue'

    # 4) fallback: first numeric column
    return nums[0] if nums else None


def detect_group_column(question: str):
    """
    Try to detect grouping column from words like:
    'by transport mode', 'by customer', 'by branch', etc.
    """
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

    # simple keyword-based mapping
    if 'transport' in q and 'TransportMode' in cats:
        return 'TransportMode'
    if 'product' in q and 'ProductLevel1' in cats:
        return 'ProductLevel1'
    if 'branch' in q and 'BranchCode' in cats:
        return 'BranchCode'
    if 'customer' in q and 'CustomerName' in cats:
        return 'CustomerName'
    if 'company' in q and 'CompanyCode' in cats:
        return 'CompanyCode'
    if 'country' in q and 'CountryName' in cats:
        return 'CountryName'

    return None


# ================================================================
# 6. Groq client
# ================================================================
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ================================================================
# 7. Main: question -> generic JSON query plan
# ================================================================
def extract_query(question: str):
    """
    Returns a *generic* query plan:

    {
      "select": [
        {"column": "Revenue", "aggregation": "sum", "alias": "value"}
      ],
      "filters": [
        {"column": "FinancialYear", "operator": "=", "value": 2024},
        {"column": "TransportMode", "operator": "in", "value": ["SEA","AIR"]}
      ],
      "group_by": ["TransportMode"],
      "order_by": [{"column": "value", "direction": "desc"}],
      "limit": 5
    }

    NOTE: "column" can be:
      - a physical column (e.g. 'REVAmount')
      - a business metric name (e.g. 'Revenue', 'ProfitPercent')
    The actual SQL expansion happens in sql_builder.py using BUSINESS_METRICS.
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

    # aggregation
    agg = "sum"
    q_lower = question.lower()
    if any(w in q_lower for w in ["average", "avg", "mean"]):
        agg = "avg"
    if any(w in q_lower for w in ["count", "how many", "number of"]):
        agg = "count"
    if any(w in q_lower for w in ["max", "maximum", "highest", "largest", "top"]):
        agg = "max"
    if any(w in q_lower for w in ["min", "minimum", "lowest", "smallest"]):
        agg = "min"

    # if metric is a business metric with a recommended default aggregation, use it
    if metric in BUSINESS_METRICS:
        default_agg = BUSINESS_METRICS[metric].get("default_agg")
        if default_agg:
            agg = default_agg

    group_col = detect_group_column(question)
    top_n = detect_top_n(question)

    base_plan = {
        "select": (
            [{"column": metric, "aggregation": agg, "alias": "value"}]
            if metric else []
        ),
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
            # Build a compact description of business metrics for the prompt
            bm_for_prompt = {
                name: {
                    "sql": meta["sql"],
                    "description": meta["description"]
                }
                for name, meta in BUSINESS_METRICS.items()
            }

            prompt = f"""
You are a senior analytics engineer.

You have:
- A physical SQL table: {TABLE_NAME}
- Physical numeric columns: {nums}
- Physical categorical columns: {cats}
- Business metrics (virtual expressions):
{json.dumps(bm_for_prompt, indent=2)}

Column descriptions (partial):
{json.dumps(COLUMN_DESCRIPTIONS, indent=2)}

TASK:
Convert the user's QUESTION into a JSON description of a SQL query.

IMPORTANT:
- When the user says "revenue", "cost", "profit %", "total job count", "total volume", etc.,
  use the BUSINESS METRIC NAME (e.g. "Revenue", "Cost", "ProfitPercent", "TotalJobCount")
  as the "column" in the select, not the raw column names.
- Use ONLY numeric columns or business metrics in "select".
- Use ONLY the listed columns in "filters" and "group_by".

JSON FORMAT (no extra text):

{{
  "select": [
    {{"column": "<numeric column OR business metric name>", "aggregation": "sum|avg|max|min|count", "alias": "value"}}
  ],
  "filters": [
    {{"column": "<column>", "operator": "=|!=|>|<|>=|<=|in|between|like", "value": "<scalar or list>"}}
  ],
  "group_by": ["<dimension columns>"],
  "order_by": [{{"column": "<column or alias>", "direction": "asc|desc"}}],
  "limit": <integer or null>
}}

Guidelines:
- If the user says "top 5", "top 10", etc., set "limit" to that integer.
- If the user asks "by transport mode / by customer / by branch / by country / by product level",
  add those columns to "group_by".
- For a single overall total, leave "group_by" empty.
- If unsure about filters, leave "filters" empty.
- Do NOT invent columns that are not in the lists.
- "column" may be a business metric name (e.g. "Revenue") or a physical numeric column.

USER QUESTION:
\"\"\"{question}\"\"\"


Return ONLY valid JSON.
"""
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512
            )
            raw = resp.choices[0].message.content.strip()
            llm_plan = json.loads(raw)

            # Merge LLM plan over fallback
            if isinstance(llm_plan, dict):
                for k, v in llm_plan.items():
                    if v not in (None, "", []):
                        base_plan[k] = v
        except Exception as e:
            print("Groq plan generation failed, using fallback:", e)

    plan = base_plan

    # --------------------
    # 3. Inject time filters (override any LLM time guesses)
    # --------------------
    filters = plan.get("filters") or []
    # Drop any FinancialYear/Quarter/Month filters the LLM may have added
    filters = [
        f for f in filters
        if str(f.get("column")) not in ("FinancialYear", "FinancialQuarter", "FinancialMonth")
    ]

    if time_ctx.get("year"):
        filters.append({
            "column": "FinancialYear",
            "operator": "=",
            "value": time_ctx["year"]
        })
    if time_ctx.get("quarter"):
        filters.append({
            "column": "FinancialQuarter",
            "operator": "=",
            "value": time_ctx["quarter"]
        })
    if time_ctx.get("month"):
        filters.append({
            "column": "FinancialMonth",
            "operator": "=",
            "value": time_ctx["month"]
        })

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
