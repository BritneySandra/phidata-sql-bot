# agent.py — CLEAN + STRICT VALUE FILTERS + stable dimension/TOP/time logic

import os
import json
import re
from datetime import datetime, timedelta
from math import ceil
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"

############################################################
# JSON LOADING
############################################################
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


############################################################
# DIMENSION MAPPING
############################################################
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

# synonyms for detecting dimension columns
DIMENSION_SYNONYMS = {
    "customer": ["customer", "client", "customer group", "customerleadgroup"],
    "branch": ["branch", "office", "location"],
    "company": ["company", "org", "organization"],
    "department": ["department", "dept", "team"],
    "job level": ["job level", "joblevel1"],
    "product": ["product", "productlevel1", "product line"],
    "job type": ["job type", "jobtype"],
    "job status": ["job status", "jobstatus"],
}

# build lookup
DIM_LOOKUP = {}
for key, col in DIMENSION_MAP.items():
    DIM_LOOKUP[key.lower()] = col
for key, syns in DIMENSION_SYNONYMS.items():
    for s in syns:
        DIM_LOOKUP[s.lower()] = DIMENSION_MAP.get(key)


############################################################
# SCHEMA LOADING
############################################################
def load_sql_schema():
    try:
        import pyodbc
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            f"Encrypt=no;TrustServerCertificate=yes;",
            timeout=5,
        )
        cur = conn.cursor()
        cur.execute(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{TABLE_NAME}'")
        rows = cur.fetchall()
        conn.close()
        return {r.COLUMN_NAME: r.DATA_TYPE.lower() for r in rows}
    except:
        return {}

_SCHEMA = {}
def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA


############################################################
# GROQ CLIENT
############################################################
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
        available = [m.id for m in client.models.list().data]
        for p in preferred:
            if p in available:
                return p
        return available[0]
    except:
        return "qwen/qwen3-32b"


############################################################
# TIME UTILS
############################################################
def calendar_to_fy_year(y, m): return y if m >= 3 else y - 1
def calendar_to_fy_month(m): return ((m - 3) % 12) + 1
def current_utc(): return datetime.utcnow()

def last_financial_quarter(ref=None):
    if not ref: ref = current_utc()
    fy = calendar_to_fy_year(ref.year, ref.month)
    fm = calendar_to_fy_month(ref.month)
    fq = ceil(fm / 3)
    return ((fy - 1, 4) if fq == 1 else (fy, fq - 1))

def previous_calendar_month(ref=None):
    if not ref: ref = current_utc()
    first = ref.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.year, prev.month

def previous_calendar_year(ref=None):
    if not ref: ref = current_utc()
    return ref.year - 1


############################################################
# TIME FILTER PARSER
############################################################
def parse_time_filters(text: str):
    q = (text or "").lower()

    # FY 2024
    m = re.search(r"\b(?:fy|financial year)\s*(20\d{2})\b", q)
    if m:
        return [{"column": "FinancialYear", "operator": "=", "value": int(m.group(1))}]

    # Quarter
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
        month = datetime.strptime(m.group(1)[:3], "%b").month
        yr = int(m.group(2))
        return [
            {"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(month)},
            {"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(yr, month)},
        ]

    # previous month
    if "previous month" in q or "last month" in q:
        y, mth = previous_calendar_month()
        return [
            {"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(mth)},
            {"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(y, mth)},
        ]

    # last quarter
    if "previous quarter" in q or "last quarter" in q:
        fy, fq = last_financial_quarter()
        return [
            {"column": "FinancialQuarter", "operator": "=", "value": fq},
            {"column": "FinancialYear", "operator": "=", "value": fy},
        ]

    # last year
    if "previous year" in q or "last year" in q:
        return [{"column": "FinancialYear", "operator": "=", "value": previous_calendar_year()}]

    # plain year
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return [{"column": "FinancialYear", "operator": "=", "value": int(m.group(1))}]

    return []


############################################################
# STRICT VALUE FILTER PARSER (NEW, FIXED)
############################################################
def extract_value_filters(question: str, schema: dict):
    """
    STRICT MODE:
    Only extract values if the user explicitly mentions the COLUMN NAME.
    Example:
        "transport mode sea" → TransportMode = 'sea'
        "sea shipments" → NO filter
    """
    q = question.lower()
    filters = []

    for col in schema.keys():
        col_low = col.lower()

        # must mention column name exactly
        if col_low not in q:
            continue

        # after the column, capture the next word as value
        m = re.search(rf"{col_low}\s*(?:is|=|:)?\s*([a-zA-Z0-9]+)", q)
        if m:
            val = m.group(1)
            filters.append({"column": col, "operator": "=", "value": val})

    return filters


############################################################
# METRIC + DIM DETECT
############################################################
DESC_KEYWORDS = {"top", "highest", "greater", "max", "most"}
ASC_KEYWORDS = {"least", "lowest", "min", "smallest", "bottom"}

def extract_top_n_and_direction(text: str):
    t = (text or "").lower()
    n = None
    m = re.search(r"top\s+(\d+)", t)
    if m: n = int(m.group(1))

    direction = None
    for w in DESC_KEYWORDS:
        if w in t: direction = "DESC"
    for w in ASC_KEYWORDS:
        if w in t: direction = "ASC"

    return n, direction, "top" in t


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
    t = text.lower()
    for syn, metric in SYNONYM_MAP.items():
        if syn in t:
            return metric
    return None


############################################################
# BUILD PLAN FOR FALLBACK METRIC
############################################################
def build_plan_from_metric(metric_key, question_text):
    m = METRICS.get(metric_key)
    if not m:
        return None

    expr = m.get("expression")
    agg = m.get("aggregation")
    alias = metric_key

    if isinstance(expr, str) and re.match(r"^\s*(sum|avg|min|max|count)\s*\(", expr.strip(), re.IGNORECASE):
        sel = {"column": None, "expression": expr, "aggregation": None, "alias": alias}
    else:
        sel = {"column": None, "expression": expr, "aggregation": agg, "alias": alias}

    plan = {"select": [sel], "filters": [], "group_by": [], "order_by": [], "limit": None}

    # add time filters
    tfs = parse_time_filters(question_text)
    for t in tfs:
        t["value"] = t.get("value")
    plan["filters"].extend(tfs)

    return plan


############################################################
# PROMPT BUILD
############################################################
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- FY begins in March; interpret plain years as FinancialYear.\n"
        "- Always output valid JSON.\n"
        "- Keys: select, filters, group_by, order_by, limit.\n"
    )
    return (
        f"You are a SQL planning engine.\n{rules}\n"
        f"QUESTION:\n{question}\n"
        f"TABLE:{TABLE_NAME}\n"
        f"SCHEMA:{json.dumps(schema)}\n"
        f"METRICS:{json.dumps(metrics)}\n"
        f"OUTPUT ONLY JSON."
    )


############################################################
# PARSE MODEL JSON OUTPUT
############################################################
def call_model_and_get_plan(client, model, prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = resp.choices[0].message.content.strip()
    s = raw.find("{")
    e = raw.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("No JSON returned")
    txt = raw[s:e+1]

    try:
        return json.loads(txt)
    except:
        try:
            return json.loads(txt.replace("'", '"'))
        except:
            raise ValueError("Invalid JSON returned")


############################################################
# NORMALIZE PLAN
############################################################
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr):
    return bool(expr and isinstance(expr, str) and AGG_RE.match(expr.strip()))


def normalize_plan(plan):
    plan = dict(plan or {})
    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    filters = plan.get("filters", []) or []
    limit = plan.get("limit")

    schema = get_schema()

    # clean group_by
    group_by = [g for g in group_by if isinstance(g, str) and g in schema]

    # ensure group_by cols in select
    clean_selects = []
    seen_alias = set()

    for g in group_by:
        clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})
        seen_alias.add(g)

    for s in selects:
        if not isinstance(s, dict):
            continue
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or col or "value"
        if is_aggregate_expression(expr):
            agg = None
        if alias not in seen_alias:
            clean_selects.append({"column": col, "expression": expr, "aggregation": agg, "alias": alias})
            seen_alias.add(alias)

    # clean order_by
    valid_alias = {s["alias"] for s in clean_selects}
    clean_order = []
    for o in order_by:
        if not isinstance(o, dict):
            continue
        col = o.get("column")
        if col in valid_alias or col in group_by:
            clean_order.append(o)

    plan["select"] = clean_selects
    plan["group_by"] = group_by
    plan["order_by"] = clean_order
    plan["filters"] = filters
    plan["limit"] = limit

    return plan


############################################################
# MAIN extract_query()
############################################################
def extract_query(question: str):
    schema = get_schema()
    client = get_client()

    prompt = build_prompt(question, schema, METADATA, METRICS)
    model = choose_best_groq_model(client) if client else None

    plan = None
    if client and model:
        try:
            plan = call_model_and_get_plan(client, model, prompt)
        except:
            plan = None

    if not plan:
        metric = detect_metric_from_text(question)
        if metric:
            plan = build_plan_from_metric(metric, question)
        else:
            return {"error": "Cannot interpret question"}

    # TIME FILTERS (authoritative)
    time_filters = parse_time_filters(question)

    # STRICT VALUE FILTERS (new)
    value_filters = extract_value_filters(question, schema)

    # merge filters
    merged_filters = list(time_filters)

    # model filters but override time filters
    time_cols = {f["column"] for f in time_filters}
    for f in plan.get("filters", []):
        if f.get("column") not in time_cols:
            merged_filters.append(f)

    # add strict value filters
    for vf in value_filters:
        if vf["column"] not in {f["column"] for f in merged_filters}:
            merged_filters.append(vf)

    plan["filters"] = merged_filters

    # TOP-N and DIMENSION logic
    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    dim_col = detect_dimension_from_text(question, schema)
    top_n, direction, has_top = extract_top_n_and_direction(question)

    # metric alias
    metric_alias = None
    for s in selects:
        if s.get("aggregation") or s.get("expression"):
            metric_alias = s.get("alias")
            break
    if not metric_alias and selects:
        metric_alias = selects[0].get("alias")

    # GROUP BY dimension
    if dim_col and dim_col not in group_by:
        group_by.insert(0, dim_col)
        selects.insert(0, {"column": dim_col, "expression": None, "aggregation": None, "alias": dim_col})

    # TOP
    if top_n and not limit:
        plan["limit"] = top_n

    # ORDER BY
    if metric_alias:
        if direction:
            order_by = [{"column": metric_alias, "direction": direction}]
        elif dim_col and not order_by:
            order_by = [{"column": metric_alias, "direction": "DESC"}]

    plan["select"] = selects
    plan["group_by"] = group_by
    plan["order_by"] = order_by

    return normalize_plan(plan)
