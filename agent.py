# agent.py — Clean, fixed, and hardened version + NEW VALUE FILTER EXTRACTION

import os
import json
import re
from datetime import datetime, timedelta
from math import ceil
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"

###############################################
# JSON LOADING
###############################################
def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

METADATA = load_json_file("metadata.json")
METRICS = load_json_file("metrics.json") or {}

# metric synonyms
SYNONYM_MAP = {}
for k, v in METRICS.items():
    syns = v.get("synonyms", []) if isinstance(v, dict) else []
    all_syns = set([k.lower()] + [s.lower() for s in syns])
    for s in all_syns:
        SYNONYM_MAP[s] = k

###############################################
# DIMENSION MAPPING
###############################################
DIMENSION_MAP = {
    "customer": "CustomerLeadGroup",
    "client": "CustomerLeadGroup",
    "customerleadgroup": "CustomerLeadGroup",
    "customer lead group": "CustomerLeadGroup",
    "customer group": "CustomerLeadGroup",
    "branch": "BranchCode",
    "office": "BranchCode",
    "location": "BranchCode",
    "company": "CompanyCode",
    "org": "CompanyCode",
    "department": "DeptCode",
    "dept": "DeptCode",
    "team": "DeptCode",
    "job level": "JobLevel1",
    "joblevel1": "JobLevel1",
    "product": "ProductLevel1",
    "productlevel1": "ProductLevel1",
    "job type": "JobType",
    "jobtype": "JobType",
    "job status": "JobStatus",
    "jobstatus": "JobStatus",
}

DIMENSION_SYNONYMS = {
    "customer": ["customer", "client", "customer group", "customerleadgroup", "client name", "clientname"],
    "branch": ["branch", "office", "location"],
    "company": ["company", "org", "organization"],
    "department": ["department", "dept", "team"],
    "job level": ["job level", "joblevel1"],
    "product": ["product", "productlevel1", "product line"],
    "job type": ["job type", "jobtype"],
    "job status": ["job status", "jobstatus"],
}

DIM_LOOKUP = {}
for key, col in DIMENSION_MAP.items():
    DIM_LOOKUP[key.lower()] = col
for key, syns in DIMENSION_SYNONYMS.items():
    for s in syns:
        DIM_LOOKUP[s.lower()] = DIMENSION_MAP.get(key)

###############################################
# SQL SCHEMA LOADING
###############################################
def load_sql_schema():
    try:
        import pyodbc
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            "Encrypt=no;TrustServerCertificate=yes;",
            timeout=5,
        )
        cursor = conn.cursor()
        cursor.execute(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{TABLE_NAME}'")
        schema = {row.COLUMN_NAME: row.DATA_TYPE.lower() for row in cursor.fetchall()}
        conn.close()
        return schema
    except:
        return {}

_SCHEMA = {}
def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA

###############################################
# GROQ CLIENT
###############################################
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except:
        return None

def choose_best_groq_model(client):
    preferred = [
        "qwen/qwen3-32b",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant",
    ]
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        for m in preferred:
            if m in available:
                return m
        return available[0]
    except:
        return "qwen/qwen3-32b"

###############################################
# TIME UTILITIES
###############################################
def calendar_to_fy_year(year, month): return year if month >= 3 else year - 1
def calendar_to_fy_month(m): return ((m - 3) % 12) + 1
def current_utc(): return datetime.utcnow()

###############################################
# TIME FILTER PARSER
###############################################
def parse_time_filters(text):
    q = (text or "").lower()

    m = re.search(r"\b(?:fy|financial year)\s*(20\d{2})\b", q)
    if m:
        return [{"column": "FinancialYear", "operator": "=", "value": int(m.group(1))}]

    m = re.search(r"\b(?:q|quarter)\s*([1-4])(?:.*?(20\d{2}))?", q)
    if m:
        qn = int(m.group(1))
        yr = int(m.group(2)) if m.group(2) else None
        out = [{"column": "FinancialQuarter", "operator": "=", "value": qn}]
        if yr: out.append({"column": "FinancialYear", "operator": "=", "value": yr})
        return out

    # Month + year
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20\d{2})\b", q)
    if m:
        mn = m.group(1)
        yr = int(m.group(2))
        cal = datetime.strptime(mn[:3], "%b").month
        return [
            {"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(cal)},
            {"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(yr, cal)},
        ]

    # previous month
    if "previous month" in q or "last month" in q:
        y, mth = previous_calendar_month()
        return [
            {"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(mth)},
            {"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(y, mth)},
        ]

    # previous quarter
    if "previous quarter" in q or "last quarter" in q:
        fy, fq = last_financial_quarter()
        return [
            {"column": "FinancialQuarter", "operator": "=", "value": fq},
            {"column": "FinancialYear", "operator": "=", "value": fy},
        ]

    # previous year
    if "previous year" in q or "last year" in q:
        return [{"column": "FinancialYear", "operator": "=", "value": previous_calendar_year()}]

    # plain year
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return [{"column": "FinancialYear", "operator": "=", "value": int(m.group(1))}]

    return []

###############################################
# NEW: GENERIC VALUE FILTER DETECTION
###############################################
def extract_value_filters(question: str, schema: dict):
    """
    Detects column-value filters from user question.
    Example: "profit for transport mode sea" ➜ TransportMode = 'sea'
    """
    q = question.lower()
    filters = []

    words = re.findall(r"[a-zA-Z0-9_]+", q)

    for col in schema.keys():
        col_low = col.lower()

        # If column name appears OR its simplified pattern appears
        if col_low in q:
            # Find nearest value after column name
            m = re.search(rf"{col_low}\s*(?:is|=|:)?\s*([a-zA-Z0-9]+)", q)
            if m:
                val = m.group(1)
                filters.append({"column": col, "operator": "=", "value": val})
                continue

        # Fallback: use ANY word that is NOT a keyword and could be a value
        for w in words:
            # ignore numeric years (already handled by time filters)
            if re.match(r"20\d\d", w):
                continue

            # If value exists in schema column (best-effort match)
            # OPTIONAL: Here you can add DB lookup for unique values
            if len(w) > 2:  # avoid small words
                filters.append({"column": col, "operator": "=", "value": w})
                break  # one value per column

    return filters


###############################################
# detect_dimension_from_text, detect_metric_from_text etc. (unchanged)
###############################################
DESC_KEYWORDS = {"top", "highest", "higher", "greater", "greatest", "max", "maximum", "most", "biggest"}
ASC_KEYWORDS = {"least", "bottom", "lowest", "lower", "minimum", "min", "smallest", "fewest"}

def extract_top_n_and_direction(text: str):
    t = (text or "").lower()
    n = None
    m = re.search(r"\btop\s+(\d+)\b", t)
    if m: n = int(m.group(1))

    direction = None
    for w in DESC_KEYWORDS:
        if w in t: direction = "DESC"
    for w in ASC_KEYWORDS:
        if w in t: direction = "ASC"

    has_top_word = "top" in t

    return n, direction, has_top_word

def detect_dimension_from_text(text: str, schema: dict):
    t = (text or "").lower()
    for col in schema.keys():
        if col.lower() in t:
            return col
    for syn, col in DIM_LOOKUP.items():
        if syn in t and col in schema:
            return col
    return None

def detect_metric_from_text(text: str):
    tl = text.lower()
    for syn, metric in SYNONYM_MAP.items():
        if syn in tl:
            return metric
    return None


###############################################
# MAIN extract_query() — with NEW FILTER SYSTEM
###############################################
def extract_query(question: str):
    schema = get_schema()
    client = get_client()

    prompt = build_prompt(question, schema, METADATA, METRICS)
    model = choose_best_groq_model(client) if client else None

    # call model
    plan = None
    if client and model:
        try:
            plan = call_model_and_get_plan(client, model, prompt)
        except:
            plan = None

    # fallback metric
    if not plan:
        metric_key = detect_metric_from_text(question)
        if metric_key:
            plan = build_plan_from_metric(metric_key, question)
        else:
            return {"error": "Could not parse question"}

    # authoritative time filters
    time_filters = parse_time_filters(question)
    for f in time_filters:
        f["value"] = sanitize_filter_value(f["value"])

    user_value_filters = extract_value_filters(question, schema)

    # merge filters
    final_filters = list(time_filters)

    for f in plan.get("filters", []):
        if f.get("column") not in {t["column"] for t in time_filters}:
            final_filters.append(f)

    # ADD NEW extracted value filters
    for vf in user_value_filters:
        if vf["column"] not in {f["column"] for f in final_filters}:
            final_filters.append(vf)

    plan["filters"] = final_filters

    # dimension/top logic
    dim_col = detect_dimension_from_text(question, schema)
    top_n, direction, has_top_word = extract_top_n_and_direction(question)

    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    # metric alias
    metric_alias = None
    for s in selects:
        if s.get("aggregation") or s.get("expression"):
            metric_alias = s.get("alias")
            break
    if not metric_alias and selects:
        metric_alias = selects[0].get("alias")

    # group by dimension
    if dim_col and dim_col not in group_by:
        group_by.insert(0, dim_col)
        selects.insert(0, {"column": dim_col, "expression": None, "aggregation": None, "alias": dim_col})

    # top N
    if top_n and not limit:
        limit = top_n
        plan["limit"] = limit

    # ordering
    if metric_alias:
        if direction:
            order_by = [{"column": metric_alias, "direction": direction}]
        elif dim_col and not order_by:
            order_by = [{"column": metric_alias, "direction": "DESC"}]

    plan["select"] = selects
    plan["group_by"] = group_by
    plan["order_by"] = order_by

    return normalize_plan(plan)
