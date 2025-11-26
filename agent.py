# agent.py — Updated: time filters authoritative, fixed filter merge, stable outputs
# plus: automatic dimension detection, TOP N, and ordering direction intelligence
import os
import json
import re
from datetime import datetime, timedelta
from math import ceil
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"


# ------------------------------------------------------
# Load JSON configuration & synonyms
# ------------------------------------------------------
def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Could not load {path}: {e}")
        return {}

METADATA = load_json_file("metadata.json")
METRICS = load_json_file("metrics.json") or {}

SYNONYM_MAP = {}
for k, v in METRICS.items():
    syns = v.get("synonyms", []) if isinstance(v, dict) else []
    all_syns = set([k.lower()] + [s.lower() for s in syns])
    for s in all_syns:
        SYNONYM_MAP[s] = k


# ------------------------------------------------------
# Dimension mapping
# ------------------------------------------------------
DIMENSION_MAP = {
    "customer": "Customerleadgroup",
    "client": "Customerleadgroup",
    "customerleadgroup": "Customerleadgroup",
    "customer group": "Customerleadgroup",
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


# ------------------------------------------------------
# Schema loader
# ------------------------------------------------------
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
        print("⚠ Could not load schema:", e)
        return {}

_SCHEMA = {}


def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA


# ------------------------------------------------------
# Groq Client + Model Selection
# ------------------------------------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        print("⚠ Could not create GROQ client:", e)
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
        for p in preferred:
            if p in available:
                return p
        for m in available:
            if "qwen" in m or "llama" in m:
                return m
        return available[0] if available else None
    except Exception:
        return "qwen/qwen3-32b"


# ------------------------------------------------------
# Time Utilities
# ------------------------------------------------------
def calendar_to_fy_year(year: int, month: int) -> int:
    return year if month >= 3 else year - 1


def month_name_to_num(name: str):
    try:
        return datetime.strptime(name[:3].capitalize(), "%b").month
    except:
        try:
            return datetime.strptime(name, "%B").month
        except:
            return None


def current_utc():
    return datetime.utcnow()


def calendar_to_fy_month(calendar_month: int) -> int:
    return ((calendar_month - 3) % 12) + 1


def fy_quarter_from_fy_month(fy_month: int) -> int:
    return ceil(fy_month / 3)


def last_financial_quarter(reference=None):
    if not reference:
        reference = current_utc()
    fy_year = calendar_to_fy_year(reference.year, reference.month)
    fy_month = calendar_to_fy_month(reference.month)
    fq = fy_quarter_from_fy_month(fy_month)
    return (fy_year - 1, 4) if fq == 1 else (fy_year, fq - 1)


def previous_calendar_month(reference=None):
    if not reference:
        reference = current_utc()
    first = reference.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.year, prev.month


def previous_calendar_year(reference=None):
    if not reference:
        reference = current_utc()
    return reference.year - 1


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def sanitize_filter_value(val):
    if isinstance(val, dict):
        for k in ("year", "quarter", "month", "value"):
            if k in val:
                return val[k]
        for v in val.values():
            if isinstance(v, int):
                return v
        return str(val)
    if isinstance(val, list):
        return val
    return val


# ------------------------------------------------------
# Time Phrase Parsing
# ------------------------------------------------------
def parse_time_filters(text: str):
    q = (text or "").lower()

    m = re.search(r"\b(?:fy|financial year)\s*[:#-]?\s*(20\d{2})\b", q)
    if m:
        return [{"column": "FinancialYear", "operator": "=", "value": int(m.group(1))}]

    m = re.search(r"\b(?:q|quarter)\s*[-:\s]*([1-4])(?:[^0-9]+(20\d{2}))?", q)
    if m:
        qnum = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else None
        out = [{"column": "FinancialQuarter", "operator": "=", "value": qnum}]
        if year:
            out.append({"column": "FinancialYear", "operator": "=", "value": year})
        return out

    m = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b",
        q,
    )
    if m:
        mon = m.group(1)
        yr = int(m.group(2))
        cal = month_name_to_num(mon)
        if cal:
            return [
                {"column": "FinancialMonth", "value": calendar_to_fy_month(cal), "operator": "="},
                {"column": "FinancialYear", "value": calendar_to_fy_year(yr, cal), "operator": "="},
            ]

    if re.search(r"\blast month\b|\bprevious month\b", q):
        y, m = previous_calendar_month()
        return [
            {"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(m)},
            {"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(y, m)},
        ]

    # FIXED LINE
    if re.search(r"\blast quarter\b|\bprevious quarter\b", q):
        fy, fq = last_financial_quarter()
        return [
            {"column": "FinancialQuarter", "value": fq, "operator": "="},
            {"column": "FinancialYear", "value": fy, "operator": "="},
        ]

    if re.search(r"\blast year\b|\bprevious year\b", q):
        return [{"column": "FinancialYear", "operator": "=", "value": previous_calendar_year()}]

    m = re.search(r"\btransaction year\s*(20\d{2})\b", q)
    if m:
        return [{"column": "TransactionYear", "operator": "=", "value": int(m.group(1))}]

    m = re.search(r"\btransaction month\s*(\d{1,2})\b", q)
    if m:
        return [{"column": "TransactionMonth", "operator": "=", "value": int(m.group(1))}]

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return [{"column": "FinancialYear", "operator": "=", "value": int(m.group(1))}]

    return []


# ------------------------------------------------------
# Plan Normalization
# ------------------------------------------------------
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)


def is_aggregate_expression(expr: str) -> bool:
    return bool(expr and isinstance(expr, str) and AGG_RE.match(expr.strip()))


def normalize_plan(plan: dict):
    plan = dict(plan or {})
    sel = plan.get("select", []) or []
    gb = plan.get("group_by", []) or []
    ob = plan.get("order_by", []) or []

    schema = get_schema()
    cleaned_gb = [g for g in gb if g and g in schema]

    cleaned_sel = []
    seen = set()

    for g in cleaned_gb:
        cleaned_sel.append({"column": g, "expression": None, "aggregation": None, "alias": g})
        seen.add(g)

    for s in sel:
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or col or "value"
        alias = alias.replace("%", "_pct")

        if expr and is_aggregate_expression(expr):
            agg = None
        if agg in ("none", "null", None):
            agg = None

        if alias not in seen:
            cleaned_sel.append({"column": col, "expression": expr, "aggregation": agg, "alias": alias})
            seen.add(alias)

    cleaned_ob = []
    sel_aliases = {s["alias"] for s in cleaned_sel}
    for o in ob:
        col = o.get("column")
        if col in sel_aliases or col in cleaned_gb:
            cleaned_ob.append(o)

    plan["select"] = cleaned_sel
    plan["group_by"] = cleaned_gb
    plan["order_by"] = cleaned_ob
    return plan


# ------------------------------------------------------
# Fallback Metric
# ------------------------------------------------------
def detect_metric_from_text(text: str):
    t = (text or "").lower()
    for syn, metric in SYNONYM_MAP.items():
        if re.search(r"\b" + re.escape(syn) + r"\b", t):
            return metric
    for syn, metric in SYNONYM_MAP.items():
        if syn in t:
            return metric
    return None


def build_plan_from_metric(metric_key, question_text):
    m = METRICS.get(metric_key)
    if not m:
        return None
    expr = m.get("expression")
    agg = m.get("aggregation")
    alias = metric_key

    if isinstance(expr, str) and re.match(r"^\s*(sum|avg|count|min|max)\s*\(", expr.strip(), re.IGNORECASE):
        sel = {"column": None, "expression": expr, "aggregation": None, "alias": alias}
    else:
        sel = {"column": None, "expression": expr, "aggregation": agg, "alias": alias}

    plan = {"select": [sel], "filters": [], "group_by": [], "order_by": []}

    tfs = parse_time_filters(question_text)
    for t in tfs:
        t["value"] = sanitize_filter_value(t["value"])
    plan["filters"].extend(tfs)

    return plan


# ------------------------------------------------------
# Prompt Builder
# ------------------------------------------------------
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- Financial Year rule: FY starts in March and ends in February next year.\n"
        "- When user mentions 'FY' or a plain 4-digit year, interpret as FinancialYear by default.\n"
        "- Use FinancialMonth/FinancialQuarter/FinancialYear columns when possible.\n"
        "- If user explicitly says 'transaction year/month', use TransactionYear/TransactionMonth.\n"
        "- Output STRICT valid JSON only.\n"
        "- Keys: select, filters, group_by, order_by, limit.\n"
        '- SELECT item format: {"column":..., "expression":..., "aggregation":..., "alias":...}\n'
    )
    return (
        f"You are a SQL planning engine.\n{rules}\nUSER QUESTION:\n{question}\nTABLE:{TABLE_NAME}\n"
        f"SCHEMA:{json.dumps(schema)}\nBUSINESS METRICS:{json.dumps(metrics)}\nOUTPUT ONLY JSON."
    )


# ------------------------------------------------------
# JSON Extractor
# ------------------------------------------------------
def call_model_and_get_plan(client, model_name, prompt):
    if not client:
        raise RuntimeError("Missing GROQ_API_KEY")

    resp = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0
    )

    raw = resp.choices[0].message.content.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("AI returned no JSON")

    txt = raw[start : end + 1]

    try:
        return json.loads(txt)
    except:
        pass

    txt2 = re.sub("'", '"', txt)
    try:
        return json.loads(txt2)
    except:
        pass

    txt3 = re.sub(r",\s*}", "}", txt)
    txt3 = re.sub(r",\s*]", "]", txt3)
    try:
        return json.loads(txt3)
    except:
        pass

    raise ValueError("Invalid JSON returned by AI")


# ------------------------------------------------------
# ------ NEW: dimension / top / direction helpers -------
# ------------------------------------------------------
DESC_KEYWORDS = {"top", "highest", "higher", "greater", "greatest", "max", "maximum", "most", "biggest"}
ASC_KEYWORDS = {"least", "bottom", "lowest", "lower", "minimum", "min", "smallest", "fewest"}


def extract_top_n_and_direction(text: str):
    t = (text or "").lower()

    m = re.search(r"\btop\s+(\d+)\b", t)
    n = int(m.group(1)) if m else None

    if n is None:
        m2 = re.search(r"\b(\d+)\s+(?:top|highest|largest|min|least)\b", t)
        if m2:
            n = int(m2.group(1))

    dir_detect = None
    has_top = False

    for w in DESC_KEYWORDS:
        if re.search(r"\b" + re.escape(w) + r"\b", t):
            dir_detect = "DESC"
            break

    for w in ASC_KEYWORDS:
        if re.search(r"\b" + re.escape(w) + r"\b", t):
            dir_detect = "ASC"
            break

    if dir_detect is None and re.search(r"\btop\b", t):
        dir_detect = "DESC"
        has_top = True

    return n, dir_detect, has_top


def detect_dimension_from_text(text: str, schema: dict):
    t = (text or "").lower()

    if schema:
        for col in schema.keys():
            if col.lower() in t:
                return col

    for syn, col in DIM_LOOKUP.items():
        if re.search(r"\b" + re.escape(syn) + r"\b", t):
            if col in schema:
                return col

    for w in re.findall(r"\w+", t):
        if w in DIM_LOOKUP and DIM_LOOKUP[w] in schema:
            return DIM_LOOKUP[w]

    return None


# ------------------------------------------------------
# MAIN extract_query()
# ------------------------------------------------------
def extract_query(question: str):
    client = get_client()
    schema = get_schema()
    prompt = build_prompt(question, schema, METADATA, METRICS)
    model = choose_best_groq_model(client)

    plan = None
    try:
        plan = call_model_and_get_plan(client, model, prompt)
    except:
        metric = detect_metric_from_text(question)
        if metric:
            plan = build_plan_from_metric(metric, question)
        else:
            return {"error": "Unparsable JSON and no metric found"}

    if not isinstance(plan, dict):
        return {"error": "AI returned non-dict"}

    time_filters = parse_time_filters(question)
    for t in time_filters:
        t["value"] = sanitize_filter_value(t["value"])

    model_filters = plan.get("filters", []) or []

    time_cols = {t["column"] for t in time_filters}
    final_filters = []
    final_filters.extend(time_filters)

    for f in model_filters:
        col = f.get("column")
        if col not in time_cols:
            f["value"] = sanitize_filter_value(f.get("value"))
            final_filters.append(f)

    plan["filters"] = final_filters

    if not plan.get("select"):
        metric = detect_metric_from_text(question)
        if metric:
            plan = build_plan_from_metric(metric, question)

    dim_col = detect_dimension_from_text(question, schema)
    top_n, direction, has_top_word = extract_top_n_and_direction(question)

    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    metric_alias = None
    for s in selects:
        if s.get("aggregation") or s.get("expression"):
            metric_alias = s.get("alias")
            break
    if not metric_alias and selects:
        metric_alias = selects[0].get("alias")

    if dim_col:
        if dim_col not in group_by:
            group_by.insert(0, dim_col)
            dim_in_select = any(
                s.get("column") == dim_col or s.get("alias") == dim_col for s in selects
            )
            if not dim_in_select:
                selects.insert(
                    0,
                    {"column": dim_col, "expression": None, "aggregation": None, "alias": dim_col},
                )

    if top_n and not limit:
        plan["limit"] = top_n
        limit = top_n

    if direction and metric_alias:
        new_order = {"column": metric_alias, "direction": direction}
        order_by = [o for o in order_by if o.get("column") != metric_alias]
        order_by.insert(0, new_order)
        plan["order_by"] = order_by

    if has_top_word and not top_n and not plan.get("limit"):
        plan["limit"] = 10

    if dim_col and not plan.get("order_by") and metric_alias:
        plan["order_by"] = [{"column": metric_alias, "direction": direction or "DESC"}]

    plan["select"] = selects
    plan["group_by"] = group_by

    return normalize_plan(plan)
