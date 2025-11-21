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

# ---------------------------------------------------
# Explicit categorical list (your confirmed list)
# ---------------------------------------------------
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
    "ConsignorShipperSuplierFullName",
]

# ---------------------------------------------------
# Load schema: COLUMN_NAME -> DATA_TYPE
# ---------------------------------------------------
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
            timeout=5,
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
        # fallback: best effort
        return {c: "varchar" for c in CATEGORICAL_COLUMNS}


_SCHEMA = {}


def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA


# ---------------------------------------------------
# Numeric / metric helpers
# ---------------------------------------------------
def numeric_columns():
    schema = get_schema()
    return [
        c
        for c, t in schema.items()
        if t in ("decimal", "numeric", "money", "float", "int", "bigint", "smallint")
    ]


METRIC_SYNONYMS = {
    # logical name  -> words the user might say
    "REVAmount": ["revenue", "rev", "sales", "turnover", "income"],
    "JobProfit": ["profit", "jobprofit", "margin", "earnings"],
    "CSTAmount": ["cost", "cst", "expense", "spend"],
}

CATEGORY_SYNONYMS = {
    "transport": ["TransportMode"],
    "transportmode": ["TransportMode"],
    "container": ["ContainerMode"],
    "product": ["ProductLevel1", "ProductLevel2", "ProductLevel3"],
    "department": ["DeptCode"],
    "branch": ["BranchCode", "AdjustedBranchCode"],
    "customer": ["CustomerName", "CustomerCode"],
    "company": ["CompanyCode"],
    "jobtype": ["JobType"],
    "joblevel1": ["JobLevel1"],
    "joblevel2": ["JobLevel2"],
    "joblevel3": ["JobLevel3"],
    "country": ["CountryName", "OriginCountry", "DestinationCountry"],
}


# ---------------------------------------------------
# Time parsing helper  (year / quarter / month)
# ---------------------------------------------------
def parse_time_from_text(question: str):
    q = question.lower()
    now = datetime.utcnow()
    res = {"year": None, "quarter": None, "month": None, "timeframe": None}

    # explicit year like 2023, 2024
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        res["year"] = int(m.group(1))

    # previous / last year
    if "previous year" in q or "last year" in q:
        res["year"] = now.year - 1
        res["timeframe"] = "previous_year"

    # last / previous quarter
    if "last quarter" in q or "previous quarter" in q:
        current_q = (now.month - 1) // 3 + 1
        prev_q = current_q - 1
        prev_year = now.year
        if prev_q == 0:
            prev_q = 4
            prev_year -= 1
        res["quarter"] = prev_q
        res["year"] = prev_year
        res["timeframe"] = "last_quarter"

    # explicit quarter like Q1 2024 / quarter 3 2024
    m = re.search(r"(?:q|quarter)[^\d]*([1-4])(?:[^0-9]+(20\d{2}))?", q)
    if m:
        res["quarter"] = int(m.group(1))
        if m.group(2):
            res["year"] = int(m.group(2))

    # last month / previous month
    if "last month" in q or "previous month" in q:
        prev = now.replace(day=1) - timedelta(days=1)
        res["month"] = prev.month
        res["year"] = prev.year
        res["timeframe"] = "last_month"

    # month name e.g. "march 2024"
    months = {
        m.lower(): i
        for i, m in enumerate(
            [
                "",
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ]
        )
    }
    for name, idx in months.items():
        if name and name in q:
            res["month"] = idx
            if not res["year"]:
                res["year"] = now.year
            break

    return res


# ---------------------------------------------------
# Metric / category detection
# ---------------------------------------------------
def find_metric_from_text(question: str):
    q = question.lower()
    schema = get_schema()

    # 1) explicit numeric column name
    for col in numeric_columns():
        if col.lower() in q:
            return col

    # 2) synonyms
    for col, syns in METRIC_SYNONYMS.items():
        if col not in schema:
            continue
        for s in syns:
            if s in q:
                return col

    # 3) heuristic fallbacks
    if "profit" in q and "JobProfit" in schema:
        return "JobProfit"
    if "revenue" in q and "REVAmount" in schema:
        return "REVAmount"

    # 4) final fallback: first numeric column
    nums = numeric_columns()
    return nums[0] if nums else None


def find_category_from_text(question: str):
    q = question.lower()
    schema = get_schema()

    # direct column mention
    for col in CATEGORICAL_COLUMNS:
        if col.lower() in q and col in schema:
            return col

    # synonyms → real columns
    for key, targets in CATEGORY_SYNONYMS.items():
        if key in q:
            for t in targets:
                if t in schema:
                    return t

    # "by <something>" pattern
    m = re.search(r"by\s+([a-z0-9 _-]{2,40})", q)
    if m:
        candidate = m.group(1).strip()
        for col in CATEGORICAL_COLUMNS:
            if candidate.replace(" ", "").lower() in col.lower() and col in schema:
                return col

    # fallback: transport
    if "transport" in q and "TransportMode" in schema:
        return "TransportMode"

    return None


def extract_category_value(question: str, category_col: str):
    """
    Try to pull an actual value for the category (e.g. SEA, 'Air Export').
    """
    q = question

    # 1) ALL CAPS codes like SEA, AIR, COU (but not years)
    codes = re.findall(r"\b[A-Z]{2,6}\b", question)
    if codes:
        filtered = [c for c in codes if not re.match(r"\d{2,4}", c)]
        if filtered:
            return filtered[0]

    # 2) Patterns around the category name
    patterns = [
        rf"{category_col.lower()}[^\w0-9]+([A-Za-z0-9 &\-/]+)",
        rf"{category_col.lower().replace('_', ' ')}[^\w0-9]+([A-Za-z0-9 &\-/]+)",
        r"for\s+([A-Za-z0-9 &\-/]+)\s+(?:in|on|for|$)",
        r"for\s+the\s+([A-Za-z0-9 &\-/]+)\s",
        r"([A-Za-z0-9 &\-/]+)\s+transport",
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            val = m.group(1).strip(" .,:;\"'")
            val = re.sub(
                r"\b(20\d{2}|last|previous|quarter|q[1-4]|in)\b.*$",
                "",
                val,
                flags=re.IGNORECASE,
            ).strip()
            if val:
                return val

    # 3) quoted phrases
    m = re.search(r"['\"]([^'\"]{2,60})['\"]", q)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------
# Groq client (optional, improves parsing)
# ---------------------------------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ---------------------------------------------------
# MAIN: question -> generic INTENT JSON
# ---------------------------------------------------
def extract_query(question: str) -> dict:
    """
    Returns a general-purpose 'intent' that sql_builder.py will convert to SQL.

    Structure:
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
      "top": 5
    }
    """

    schema = get_schema()
    q_lower = question.lower()

    # ---- metric, agg, time ----
    metric = find_metric_from_text(question)
    time = parse_time_from_text(question)

    # aggregation
    agg = "sum"
    if any(w in q_lower for w in ["average", "avg", "mean"]):
        agg = "avg"
    if any(w in q_lower for w in ["count", "how many", "number of"]):
        agg = "count"
    if any(w in q_lower for w in ["max", "maximum", "largest", "highest"]):
        agg = "max"
    if any(w in q_lower for w in ["min", "minimum", "smallest", "lowest"]):
        agg = "min"

    # ---- category & grouping ----
    category_col = find_category_from_text(question)
    category_value = None

    if category_col:
        category_value = extract_category_value(question, category_col)
        if category_value and re.fullmatch(r"[A-Za-z]{2,6}", category_value.strip()):
            category_value = category_value.strip().upper()
        elif category_value:
            category_value = category_value.strip()

    # group-by intent
    group_by_cols = []
    group_words = [
        " by ",
        " breakdown",
        " group ",
        " split ",
        " per ",
        " distribution",
        " across ",
    ]
    has_group_word = any(w in q_lower for w in group_words)

    if category_col and has_group_word and not category_value:
        # e.g. "revenue by transport mode"
        group_by_cols = [category_col]

    # ---- compare values (for IN filter) ----
    compare_values = []
    if "compare" in q_lower or " vs " in q_lower or " versus " in q_lower:
        parts = re.split(r"compare|versus| vs | vs\.|,| and | & ", question, flags=re.IGNORECASE)
        for part in parts[1:]:
            tokens = re.findall(r"[A-Za-z0-9 &\-/]{1,40}", part)
            for t in tokens:
                t = t.strip()
                if t and not re.search(
                    r"\b(last|previous|year|quarter|q[1-4])\b", t, re.IGNORECASE
                ):
                    compare_values.append(t)
        compare_values = [c for i, c in enumerate(compare_values) if c and c not in compare_values[:i]]

    # ---- top / bottom N ----
    top_n = None
    order_direction = None

    m_top = re.search(r"\btop\s+(\d+)\b", q_lower)
    m_bottom = re.search(r"\bbottom\s+(\d+)\b", q_lower)

    if m_top:
        top_n = int(m_top.group(1))
        order_direction = "desc"
    elif m_bottom:
        top_n = int(m_bottom.group(1))
        order_direction = "asc"

    # ---- build filters list ----
    filters = []

    if time.get("year") is not None:
        filters.append(
            {"column": "FinancialYear", "operator": "=", "value": time["year"]}
        )
    if time.get("quarter") is not None:
        filters.append(
            {
                "column": "FinancialQuarter",
                "operator": "=",
                "value": time["quarter"],
            }
        )
    if time.get("month") is not None:
        filters.append(
            {"column": "FinancialMonth", "operator": "=", "value": time["month"]}
        )

    if category_col and category_value:
        filters.append(
            {
                "column": category_col,
                "operator": "=",
                "value": category_value,
            }
        )
    elif category_col and compare_values:
        filters.append(
            {
                "column": category_col,
                "operator": "in",
                "value": compare_values,
            }
        )

    # ---- ORDER BY ----
    order_by = []
    if metric:
        metric_alias = "value"
        if group_by_cols:
            direction = order_direction or ("desc" if agg in ("sum", "avg", "max") else "asc")
            order_by.append({"column": metric_alias, "direction": direction})

    # ---- SELECT list ----
    select = []
    if metric:
        select.append(
            {"column": metric, "aggregation": agg, "alias": "value"}
        )

    base_intent = {
        "select": select,
        "filters": filters,
        "group_by": group_by_cols,
        "order_by": order_by,
        "top": top_n,
    }

    # ------------------------------------------------
    # Optional: ask Groq to refine / fix the intent
    # ------------------------------------------------
    client = get_client()
    if not client:
        return base_intent

    try:
        prompt = f"""
You are a JSON-only assistant.

You are given a list of SQL columns:
{list(schema.keys())}

Convert the user question into this JSON format:

{{
  "select": [
    {{"column": "<numeric column>", "aggregation": "sum|avg|max|min|count", "alias": "value"}}
  ],
  "filters": [
    {{"column": "<column>", "operator": "=|>|<|>=|<=|<>|in|between", "value": <value or [values]>}}
  ],
  "group_by": ["<column>", "..."],
  "order_by": [
    {{"column": "<column or alias>", "direction": "asc|desc"}}
  ],
  "top": <integer or null>
}}

Rules:
- Use only column names that exist in the provided list.
- If the user says "top N", set "top" to N and sort by the aggregated metric alias "value" desc.
- If the user filters by a category value (e.g. SEA, Air Export), use a filter with operator "=".
- If the user compares values (SEA vs AIR), use operator "in" with a list of values.
- If no specific time is mentioned, leave time filters out.

User question: "{question}"

Return ONLY JSON. No extra text.
"""
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        llm_intent = json.loads(raw)

        # ------ merge LLM intent on top of base_intent, with validation ------
        intent = base_intent

        # select
        sel = llm_intent.get("select")
        if isinstance(sel, list) and sel:
            clean_sel = []
            for s in sel:
                col = s.get("column")
                if col and col in schema and col in numeric_columns():
                    clean_sel.append(
                        {
                            "column": col,
                            "aggregation": s.get("aggregation", "sum"),
                            "alias": s.get("alias", "value"),
                        }
                    )
            if clean_sel:
                intent["select"] = clean_sel

        # filters
        llm_filters = llm_intent.get("filters") or []
        clean_filters = []
        for f in llm_filters:
            col = f.get("column")
            if col and col in schema:
                clean_filters.append(f)
        if clean_filters:
            intent["filters"] = clean_filters

        # group_by
        llm_group = llm_intent.get("group_by") or []
        clean_group = [c for c in llm_group if c in schema]
        if clean_group:
            intent["group_by"] = clean_group

        # order_by
        llm_order = llm_intent.get("order_by") or []
        if llm_order:
            intent["order_by"] = llm_order

        # top
        if llm_intent.get("top") is not None:
            try:
                intent["top"] = int(llm_intent["top"])
            except Exception:
                pass

        return intent

    except Exception as e:
        print("LLM intent parse failed:", e)
        return base_intent
