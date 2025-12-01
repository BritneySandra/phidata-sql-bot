# agent.py — CLEAN + STRICT VALUE FILTERS + stable dimension/TOP/time logic
# Updated with:
# 1) FORCE METRIC DETECTED FROM QUESTION
# 2) Numeric-limit patch inserted IMMEDIATELY AFTER extract_top_n_and_direction()
# 3) YOY detection: when user asks to compare two years (or previous/this), force FinancialYear IN (...) and group_by FinancialYear
# 4) MOM detection (previous month / explicit month pairs) — add FinancialMonth + FinancialYear IN filters and group_by
# 5) QoQ detection (previous quarter / explicit quarter pairs) — add FinancialQuarter + FinancialYear IN filters and group_by
# Minimal, focused changes only — rest of your working logic preserved.

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
METRICS  = load_json_file("metrics.json") or {}

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
    # transport additions
    "transport": "TransportMode",
    "transport mode": "TransportMode",
    "mode": "TransportMode",
    "mode of transport": "TransportMode",
}

DIMENSION_SYNONYMS = {
    "customer": ["customer", "client", "customer group", "customerleadgroup"],
    "branch": ["branch", "office", "location"],
    "company": ["company", "org", "organization"],
    "department": ["department", "dept", "team"],
    "job level": ["job level", "joblevel1"],
    "product": ["product", "productlevel1", "product line"],
    "job type": ["job type", "jobtype"],
    "job status": ["job status", "jobstatus"],
    "transport": ["transport", "transport mode", "mode of transport", "mode"],
}

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

    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20\d{2})\b", q)
    if m:
        month = datetime.strptime(m.group(1)[:3], "%b").month
        yr = int(m.group(2))
        return [
            {"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(month)},
            {"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(yr, month)},
        ]

    if "previous month" in q or "last month" in q:
        y, mth = previous_calendar_month()
        return [
            {"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(mth)},
            {"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(y, mth)},
        ]

    if "previous quarter" in q or "last quarter" in q:
        fy, fq = last_financial_quarter()
        return [
            {"column": "FinancialQuarter", "operator": "=", "value": fq},
            {"column": "FinancialYear", "operator": "=", "value": fy},
        ]

    if "previous year" in q or "last year" in q:
        return [{"column": "FinancialYear", "operator": "=", "value": previous_calendar_year()}]

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return [{"column": "FinancialYear", "operator": "=", "value": int(m.group(1))}]

    return []

############################################################
# STRICT VALUE FILTER PARSER
############################################################
def extract_value_filters(question: str, schema: dict):
    q = question.lower()
    filters = []

    if not schema:
        return filters

    for col in schema.keys():
        col_low = col.lower()
        if col_low not in q:
            continue

        m = re.search(rf"{re.escape(col_low)}\s*(?:is|=|:)?\s*([a-zA-Z0-9\-_]+)", q)
        if m:
            filters.append({"column": col, "operator": "=", "value": m.group(1)})

    return filters

############################################################
# METRIC + DIM DETECT
############################################################
DESC_KEYWORDS = {"top", "highest", "greater", "max", "most", "biggest", "maximum"}
ASC_KEYWORDS = {"least", "lowest", "min", "smallest", "bottom", "minimum"}

def extract_top_n_and_direction(text: str):
    t = (text or "").lower()
    n = None

    m = re.search(r"\btop\s+(\d+)\b", t)
    if m:
        n = int(m.group(1))

    direction = None
    for w in DESC_KEYWORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            direction = "DESC"
            break
    for w in ASC_KEYWORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            direction = "ASC"
            break

    return n, direction, "top" in t


def detect_dimension_from_text(text: str, schema: dict):
    t = (text or "").lower()

    if schema:
        for col in schema.keys():
            if col.lower() in t:
                return col

    for syn, col in DIM_LOOKUP.items():
        if syn in t:
            if not schema or col in schema:
                return col

    return None


def detect_metric_from_text(text: str):
    """
    Robust metric detection:
    - Prefer explicit percent requests to map to *_percentage metrics (e.g., profit_percentage).
    - Use whole-word matching for synonyms (avoid substring matches).
    - Prefer longer synonym matches first (to avoid 'rev' matching 'revenue' incorrectly).
    - Fall back to any synonym match otherwise.
    """
    t = (text or "").lower()

    # Quick empty guard
    if not t or not SYNONYM_MAP:
        return None

    # 1) If question explicitly requests percentage, prefer *_percentage metrics.
    if "%" in t or " percent" in t or "percentage" in t:
        # If a profit percentage metric exists, and the user mentions profit, prefer it
        if "profit_percentage" in METRICS and re.search(r"\bprofit\b", t):
            return "profit_percentage"
        # Generic: look for any metric whose key name contains 'percentage' or synonyms that contain 'percent'
        for key, meta in METRICS.items():
            if key.lower().endswith("percentage") or "percentage" in key.lower() or "percent" in key.lower():
                # check if any of its synonyms appear as whole words in the question
                syns = meta.get("synonyms", []) if isinstance(meta, dict) else []
                for s in ([key] + syns):
                    if re.search(rf"\b{re.escape(s.lower())}\b", t):
                        return key

    # 2) Priority explicit percent phrases (e.g. "profit %", "profit percent", "margin %")
    if re.search(r"\b(profit\s*%|profit percent|profit percentage|margin\s*%|margin percent|margin percentage)\b", t):
        if "profit_percentage" in METRICS:
            return "profit_percentage"

    # 3) Whole-word synonym matching (longer synonyms first to avoid substring issues)
    syn_list = sorted(SYNONYM_MAP.keys(), key=lambda s: len(s), reverse=True)
    for syn in syn_list:
        # build whole-word regex (handles multi-word synonyms)
        pattern = rf"\b{re.escape(syn)}\b"
        if re.search(pattern, t):
            return SYNONYM_MAP[syn]

    # 4) As a last resort, try simple contains (fallback)
    for syn, metric in SYNONYM_MAP.items():
        if syn in t:
            return metric

    return None

############################################################
# FALLBACK METRIC PLAN
############################################################
def build_plan_from_metric(metric_key, question_text):
    m = METRICS.get(metric_key)
    if not m:
        return None

    expr = m.get("expression")
    agg  = m.get("aggregation")
    alias = metric_key

    if isinstance(expr, str) and re.match(r"^\s*(sum|avg|min|max|count)\s*\(", expr.strip(), re.IGNORECASE):
        sel = {"column": None, "expression": expr, "aggregation": None, "alias": alias}
    else:
        sel = {"column": None, "expression": expr, "aggregation": agg, "alias": alias}

    plan = {"select": [sel], "filters": [], "group_by": [], "order_by": [], "limit": None}

    tfs = parse_time_filters(question_text)
    plan["filters"].extend(tfs)

    return plan

############################################################
# LLM PROMPT
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
# CALL MODEL
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
        return json.loads(txt.replace("'", '"'))

############################################################
# NORMALIZE PLAN
############################################################
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)
def is_aggregate_expression(expr):
    return bool(expr and isinstance(expr, str) and AGG_RE.match(expr.strip()))

def normalize_plan(plan):
    plan = dict(plan or {})
    selects  = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    filters  = plan.get("filters", []) or []
    limit    = plan.get("limit")

    schema = get_schema() or {}

    gb_clean = []
    for g in group_by:
        if isinstance(g, str) and (not schema or g in schema):
            gb_clean.append(g)

    clean_selects = []
    seen_alias = set()

    for g in gb_clean:
        clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})
        seen_alias.add(g)

    for s in selects:
        if not isinstance(s, dict):
            continue

        col   = s.get("column")
        expr  = s.get("expression")
        agg   = s.get("aggregation")
        alias = s.get("alias") or col or "value"

        if is_aggregate_expression(expr):
            agg = None

        if alias not in seen_alias:
            clean_selects.append({"column": col, "expression": expr, "aggregation": agg, "alias": alias})
            seen_alias.add(alias)

    valid_aliases = {s["alias"] for s in clean_selects}
    clean_order = []
    for o in order_by:
        if not isinstance(o, dict):
            continue

        col = o.get("column")
        direction = o.get("direction", "DESC") or "DESC"

        if col in valid_aliases or col in gb_clean:
            clean_order.append({"column": col, "direction": direction})

    plan["select"]  = clean_selects
    plan["group_by"] = gb_clean
    plan["order_by"] = clean_order
    plan["filters"] = filters
    plan["limit"]   = limit

    return plan

############################################################
# COERCE PLAN SHAPES
############################################################
def coerce_plan_shapes(plan):
    if not isinstance(plan, dict):
        return plan

    s = plan.get("select", [])
    if s is None: s = []
    coerced_selects = []
    for item in s:
        if isinstance(item, dict):
            coerced_selects.append(item)
        elif isinstance(item, str):
            coerced_selects.append({"column": None, "expression": None, "aggregation": None, "alias": item})
    plan["select"] = coerced_selects

    f = plan.get("filters", [])
    if f is None: f = []
    plan["filters"] = [x for x in f if isinstance(x, dict)]

    ob = plan.get("order_by", [])
    if ob is None: ob = []
    coerced_ob = []
    for item in ob:
        if isinstance(item, dict):
            coerced_ob.append(item)
        elif isinstance(item, str):
            coerced_ob.append({"column": item, "direction": "DESC"})
    plan["order_by"] = coerced_ob

    gb = plan.get("group_by", []) or []
    coerced_gb = []
    for item in gb:
        if isinstance(item, str):
            coerced_gb.append(item)
        elif isinstance(item, dict):
            col = item.get("column")
            if isinstance(col, str):
                coerced_gb.append(col)
    plan["group_by"] = coerced_gb

    return plan

############################################################
# MAIN extract_query()
############################################################
def extract_query(question: str):
    schema = get_schema()
    client = get_client()

    prompt = build_prompt(question, schema, METADATA, METRICS)
    model  = choose_best_groq_model(client) if client else None

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
            return {"error": "Cannot interpret question (no model output and no metric detected)"}

    plan = coerce_plan_shapes(plan or {})

    # parse time and value filters (initial)
    time_filters  = parse_time_filters(question)
    value_filters = extract_value_filters(question, schema)

    merged_filters = list(time_filters)
    time_cols = {f.get("column") for f in time_filters}

    for f in plan.get("filters", []):
        if isinstance(f, dict) and f.get("column") not in time_cols:
            merged_filters.append(f)

    existing_cols = {f.get("column") for f in merged_filters}
    for vf in value_filters:
        if vf.get("column") not in existing_cols:
            merged_filters.append(vf)

    # --- YOY detection: look for explicit two-year comparisons or "previous year" vs "this year" ---
    # If detected, replace/add time filter as an IN filter with both years and ensure FinancialYear is grouped/selected.
    yoy_years = None
    qlow = (question or "").lower()

    # 1) explicit pairs like "compare 2023 and 2024" or "2024 vs 2023"
    m_pair = re.search(r"\b(20\d{2})\b.*\b(?:and|vs|v|vs\.|versus|to)\b.*\b(20\d{2})\b", qlow)
    if m_pair:
        y1 = int(m_pair.group(1))
        y2 = int(m_pair.group(2))
        # normalize order (ascending)
        yoy_years = sorted({y1, y2})

    # 2) "compare previous year and this year" or "compare previous year with this year"
    if not yoy_years:
        if re.search(r"\b(previous year|last year)\b", qlow) and re.search(r"\b(this year|current year|present year)\b", qlow):
            this_y = current_utc().year
            prev_y = this_y - 1
            yoy_years = [prev_y, this_y]

    # 3) "compare 2023 and last year" or "compare 2024 and previous year" (mixed)
    if not yoy_years:
        m_mixed = re.search(r"\b(20\d{2})\b.*\b(?:and|with|vs|versus)\b.*\b(previous year|last year|this year|current year)\b", qlow)
        if m_mixed:
            y_fixed = int(m_mixed.group(1))
            if re.search(r"\b(previous year|last year)\b", qlow):
                yoy_years = sorted({y_fixed, y_fixed - 1})
            else:
                this_y = current_utc().year
                yoy_years = sorted({y_fixed, this_y})

    # If we detected a YOY comparison, adjust merged_filters to include FinancialYear IN [years]
    if yoy_years:
        # Remove any existing FinancialYear equality filters from merged_filters
        merged_filters = [f for f in merged_filters if not (f.get("column") == "FinancialYear" and f.get("operator") == "=")]
        # Add an IN filter for the years (sql_builder supports operator 'in' with list)
        merged_filters.append({"column": "FinancialYear", "operator": "in", "value": yoy_years})
        # Ensure FinancialYear is in group_by so output is per-year
        gb = plan.get("group_by", []) or []
        if "FinancialYear" not in gb:
            gb.insert(0, "FinancialYear")
            plan["group_by"] = gb
        # Ensure FinancialYear is selected
        selects = plan.get("select", []) or []
        if not any(isinstance(s, dict) and (s.get("column") == "FinancialYear" or s.get("alias") == "FinancialYear") for s in selects):
            selects.insert(0, {"column": "FinancialYear", "expression": None, "aggregation": None, "alias": "FinancialYear"})
            plan["select"] = selects

    # --- MOM detection (month-over-month) ---
    # Support:
    #  - "previous month and this month" / "compare previous month and this month"
    #  - explicit pair: "Jan 2024 and Feb 2024" / "Jan 2024 vs Feb 2024"
    mom_months = None
    if not yoy_years:  # prefer YOY if that matched; otherwise MOM allowed
        # 1) "previous month" and "this month"
        if re.search(r"\b(previous month|last month)\b", qlow) and re.search(r"\b(this month|current month|present month)\b", qlow):
            this_dt = current_utc()
            prev_dt = (this_dt.replace(day=1) - timedelta(days=1))
            this_fy = calendar_to_fy_year(this_dt.year, this_dt.month)
            this_fm = calendar_to_fy_month(this_dt.month)
            prev_fy = calendar_to_fy_year(prev_dt.year, prev_dt.month)
            prev_fm = calendar_to_fy_month(prev_dt.month)
            mom_months = [(prev_fy, prev_fm), (this_fy, this_fm)]

        # 2) explicit month pairs like "Jan 2024 and Feb 2024" or "Jan 2023 vs Feb 2023"
        if not mom_months:
            m_month_pair = re.search(
                r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20\d{2})\b.*\b(?:and|vs|v|versus|to)\b.*\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20\d{2})\b",
                qlow
            )
            if m_month_pair:
                m1 = datetime.strptime(m_month_pair.group(1)[:3], "%b").month
                y1 = int(m_month_pair.group(2))
                m2 = datetime.strptime(m_month_pair.group(3)[:3], "%b").month
                y2 = int(m_month_pair.group(4))
                fy1 = calendar_to_fy_year(y1, m1)
                fy2 = calendar_to_fy_year(y2, m2)
                fm1 = calendar_to_fy_month(m1)
                fm2 = calendar_to_fy_month(m2)
                mom_months = [(fy1, fm1), (fy2, fm2)]

    if mom_months:
        # Remove individual FinancialMonth / FinancialYear equality filters if present
        merged_filters = [f for f in merged_filters if not (f.get("column") in ("FinancialMonth", "FinancialYear") and f.get("operator") == "=")]
        # Build IN lists for months and years (main.py can detect these IN filters)
        years = sorted({p[0] for p in mom_months})
        months = sorted({p[1] for p in mom_months})
        merged_filters.append({"column": "FinancialYear", "operator": "in", "value": years})
        merged_filters.append({"column": "FinancialMonth", "operator": "in", "value": months})
        # Ensure grouping by FinancialYear and FinancialMonth
        gb = plan.get("group_by", []) or []
        if "FinancialYear" not in gb:
            gb.insert(0, "FinancialYear")
        if "FinancialMonth" not in gb:
            # place month after year
            gb.insert(1 if "FinancialYear" in gb else 0, "FinancialMonth")
        plan["group_by"] = gb
        # Ensure selection includes both
        selects = plan.get("select", []) or []
        if not any(isinstance(s, dict) and (s.get("column") == "FinancialYear" or s.get("alias") == "FinancialYear") for s in selects):
            selects.insert(0, {"column": "FinancialYear", "expression": None, "aggregation": None, "alias": "FinancialYear"})
        if not any(isinstance(s, dict) and (s.get("column") == "FinancialMonth" or s.get("alias") == "FinancialMonth") for s in selects):
            # insert after FinancialYear
            inserts_at = 1 if any(isinstance(s, dict) and s.get("alias") == "FinancialYear" for s in selects) else 0
            selects.insert(inserts_at, {"column": "FinancialMonth", "expression": None, "aggregation": None, "alias": "FinancialMonth"})
        plan["select"] = selects

    # --- QoQ detection (quarter-over-quarter) ---
    # Support:
    #  - "previous quarter and this quarter"
    #  - explicit quarter pairs like "Q1 2023 and Q2 2023"
    qoq_quarters = None
    if not (yoy_years or mom_months):
        # 1) previous quarter and this quarter
        if re.search(r"\b(previous quarter|last quarter)\b", qlow) and re.search(r"\b(this quarter|current quarter|present quarter)\b", qlow):
            this_dt = current_utc()
            fy_this = calendar_to_fy_year(this_dt.year, this_dt.month)
            fm_this = calendar_to_fy_month(this_dt.month)
            q_this = ceil(fm_this/3)
            # compute previous quarter fiscal tuple
            if q_this == 1:
                q_prev = 4
                fy_prev = fy_this - 1
            else:
                q_prev = q_this - 1
                fy_prev = fy_this
            qoq_quarters = [(fy_prev, q_prev), (fy_this, q_this)]

        # 2) explicit quarter pairs like "Q1 2023 and Q2 2023"
        if not qoq_quarters:
            m_q_pair = re.search(r"\bq([1-4])\s*(20\d{2})\b.*\b(?:and|vs|v|versus|to)\b.*\bq([1-4])\s*(20\d{2})\b", qlow)
            if m_q_pair:
                q1 = int(m_q_pair.group(1))
                y1 = int(m_q_pair.group(2))
                q2 = int(m_q_pair.group(3))
                y2 = int(m_q_pair.group(4))
                fy1 = calendar_to_fy_year(y1, (q1-1)*3 + 3)  # approximate month -> fy
                fy2 = calendar_to_fy_year(y2, (q2-1)*3 + 3)
                qoq_quarters = [(fy1, q1), (fy2, q2)]

    if qoq_quarters:
        # Remove any existing FinancialQuarter / FinancialYear equality filters
        merged_filters = [f for f in merged_filters if not (f.get("column") in ("FinancialQuarter", "FinancialYear") and f.get("operator") == "=")]
        years = sorted({p[0] for p in qoq_quarters})
        quarters = sorted({p[1] for p in qoq_quarters})
        merged_filters.append({"column": "FinancialYear", "operator": "in", "value": years})
        merged_filters.append({"column": "FinancialQuarter", "operator": "in", "value": quarters})
        # Ensure grouping by FinancialYear and FinancialQuarter
        gb = plan.get("group_by", []) or []
        if "FinancialYear" not in gb:
            gb.insert(0, "FinancialYear")
        if "FinancialQuarter" not in gb:
            gb.insert(1 if "FinancialYear" in gb else 0, "FinancialQuarter")
        plan["group_by"] = gb
        # Ensure selection includes both
        selects = plan.get("select", []) or []
        if not any(isinstance(s, dict) and (s.get("column") == "FinancialYear" or s.get("alias") == "FinancialYear") for s in selects):
            selects.insert(0, {"column": "FinancialYear", "expression": None, "aggregation": None, "alias": "FinancialYear"})
        if not any(isinstance(s, dict) and (s.get("column") == "FinancialQuarter" or s.get("alias") == "FinancialQuarter") for s in selects):
            inserts_at = 1 if any(isinstance(s, dict) and s.get("alias") == "FinancialYear" for s in selects) else 0
            selects.insert(inserts_at, {"column": "FinancialQuarter", "expression": None, "aggregation": None, "alias": "FinancialQuarter"})
        plan["select"] = selects

    plan["filters"] = merged_filters

    # Now proceed with your existing select/group_by/top logic
    selects  = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit    = plan.get("limit")

    #########################################################
    # 🔥 FORCE METRIC DETECTED FROM QUESTION
    #########################################################
    metric_key = detect_metric_from_text(question)
    if metric_key and metric_key in METRICS:
        metric_info = METRICS[metric_key]
        expr  = metric_info.get("expression")
        agg   = metric_info.get("aggregation")
        alias = metric_key

        already = any(
            isinstance(s, dict) and s.get("alias") == alias
            for s in selects
        )
        if not already:
            selects.append({
                "column": None,
                "expression": expr,
                "aggregation": agg,
                "alias": alias
            })

    plan["select"] = selects

    #########################################################
    # DIMENSION + TOP DETECT
    #########################################################
    dim_col = detect_dimension_from_text(question, schema)
    top_n, direction, has_top = extract_top_n_and_direction(question)

    #########################################################
    # 🔥 NUMERIC-LIMIT PATCH (YEAR-SAFE)
    # Only treat small numbers (1–100) as TOP limits.
    # Ignore years (1900–2099)
    #########################################################
    explicit_n = None
    if not top_n:  # Only override when LLM did NOT already set a top_n
        m = re.search(r"\b(\d+)\b", question)
        if m:
            num = int(m.group(1))
            # treat typical top-n values as limits, ignore likely years
            if 1 <= num <= 100:
                explicit_n = num

    if explicit_n:
        plan["limit"] = explicit_n
        top_n = explicit_n
    #########################################################

    if dim_col:
        for col in schema.keys():
            if col.lower() == dim_col.lower():
                dim_col = col
                break

        if dim_col not in group_by:
            group_by.insert(0, dim_col)

        dim_present = any(
            isinstance(s, dict) and (s.get("alias") == dim_col or s.get("column") == dim_col)
            for s in selects
        )
        if not dim_present:
            selects.insert(0, {
                "column": dim_col,
                "expression": None,
                "aggregation": None,
                "alias": dim_col
            })

    if top_n:
        plan["limit"] = top_n

    metric_alias = None
    for s in selects:
        if isinstance(s, dict) and (s.get("aggregation") or s.get("expression")):
            metric_alias = s.get("alias")
            break

    if metric_alias:
        if direction:
            order_by = [{"column": metric_alias, "direction": direction}]
        elif dim_col and not order_by:
            order_by = [{"column": metric_alias, "direction": "DESC"}]

    plan["select"]  = selects
    plan["group_by"] = group_by
    plan["order_by"] = order_by

    plan = normalize_plan(plan)

    valid = []
    for s in plan.get("select", []):
        if isinstance(s, dict):
            c = s.get("column")
            e = s.get("expression")
            if (c and c.strip()) or (e and e.strip()):
                valid.append(s)

    if not valid:
        metric = detect_metric_from_text(question)
        if metric:
            return build_plan_from_metric(metric, question)
        return {"error": "No valid SELECT expressions even after metric injection"}

    # ---------------------------------------------------------
    # FINAL METRIC INJECTION (AFTER NORMALIZE)
    # ---------------------------------------------------------
    metric_key = detect_metric_from_text(question)
    if metric_key and metric_key in METRICS:
        metric_info = METRICS[metric_key]

        expr  = metric_info.get("expression")
        agg   = metric_info.get("aggregation")
        alias = metric_key

        # Check if already exists
        already = any(
            isinstance(s, dict) and s.get("alias") == alias
            for s in plan["select"]
        )

        if not already:
            plan["select"].append({
                "column": None,
                "expression": expr,
                "aggregation": agg,
                "alias": alias
            })

    return plan
