# agent.py — Final merged, cleaned, and working version
# Features:
# - Robust time filters (FY starts in March)
# - Dimension detection (Customer -> CustomerLeadGroup)
# - Top-N & ordering direction detection
# - Metric fallback when LLM plan missing metric
# - Value filter extraction (e.g., "transport mode sea")
# - Robust LLM JSON parsing and plan normalization
# - Safe schema loading and resilient to missing GROQ key

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
    except Exception:
        return {}

METADATA = load_json_file("metadata.json")
METRICS = load_json_file("metrics.json") or {}

# metric synonyms map
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
        conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            f"Encrypt=no;TrustServerCertificate=yes;"
        )
        conn = pyodbc.connect(conn_str, timeout=5)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?",
            (TABLE_NAME,),
        )
        schema = {row.COLUMN_NAME: row.DATA_TYPE.lower() for row in cursor.fetchall()}
        conn.close()
        return schema
    except Exception as e:
        # don't crash when schema not available (useful for dev)
        print("⚠ Could not load schema:", e)
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
        # fallback to first available
        return available[0] if available else None
    except Exception:
        return "qwen/qwen3-32b"

###############################################
# TIME UTILITIES
###############################################
def calendar_to_fy_year(year, month): return year if month >= 3 else year - 1
def calendar_to_fy_month(m): return ((m - 3) % 12) + 1
def current_utc(): return datetime.utcnow()
def fy_quarter_from_fy_month(fy_month): return ceil(fy_month / 3)

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

###############################################
# HELPERS & SANITIZERS
###############################################
def sanitize_filter_value(val):
    if isinstance(val, dict):
        for k in ("year","quarter","month","value"):
            if k in val:
                return val[k]
        for v in val.values():
            if isinstance(v, int):
                return v
        return str(val)
    if isinstance(val, list):
        return val
    return val

###############################################
# TIME FILTER PARSER
###############################################
def parse_time_filters(text):
    q = (text or "").lower()
    filters = []

    m = re.search(r'\b(?:fy|financial year)\s*[:#-]?\s*(20\d{2})\b', q)
    if m:
        filters.append({"column":"FinancialYear","operator":"=","value": int(m.group(1))})
        return filters

    m = re.search(r'\b(?:q|quarter)\s*[-:\s]*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        qnum = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else None
        filters.append({"column":"FinancialQuarter","operator":"=","value": qnum})
        if year:
            filters.append({"column":"FinancialYear","operator":"=","value": year})
        return filters

    m = re.search(r'\b('
                  r'january|february|march|april|may|june|july|august|september|october|november|december|'
                  r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
                  r')\s+(20\d{2})\b', q, flags=re.IGNORECASE)
    if m:
        mon = m.group(1)
        yr = int(m.group(2))
        try:
            cal_month = datetime.strptime(mon[:3].capitalize(), "%b").month
            fy_month = calendar_to_fy_month(cal_month)
            fy_year = calendar_to_fy_year(yr, cal_month)
            filters.append({"column":"FinancialMonth","operator":"=","value": fy_month})
            filters.append({"column":"FinancialYear","operator":"=","value": fy_year})
            return filters
        except Exception:
            pass

    if re.search(r'\blast month\b|\bprevious month\b', q):
        y, m = previous_calendar_month()
        filters.append({"column":"FinancialMonth","operator":"=","value": calendar_to_fy_month(m)})
        filters.append({"column":"FinancialYear","operator":"=","value": calendar_to_fy_year(y, m)})
        return filters

    if re.search(r'\blast quarter\b|\bprevious quarter\b', q):
        fy, fq = last_financial_quarter()
        filters.append({"column":"FinancialQuarter","operator":"=","value": fq})
        filters.append({"column":"FinancialYear","operator":"=","value": fy})
        return filters

    if re.search(r'\blast year\b|\bprevious year\b', q):
        filters.append({"column":"FinancialYear","operator":"=","value": previous_calendar_year()})
        return filters

    m = re.search(r'\btransaction year\s*(20\d{2})\b', q)
    if m:
        filters.append({"column":"TransactionYear","operator":"=","value": int(m.group(1))})
        return filters
    m = re.search(r'\btransaction month\s*(\d{1,2})\b', q)
    if m:
        filters.append({"column":"TransactionMonth","operator":"=","value": int(m.group(1))})
        return filters

    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        filters.append({"column":"FinancialYear","operator":"=","value": int(m.group(1))})
        return filters

    return []

###############################################
# ROBUST LLM JSON PARSING
###############################################
def call_model_and_get_plan(client, model_name, prompt):
    if client is None:
        raise RuntimeError("Missing GROQ_API_KEY or unable to create Groq client")

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
    except Exception as e:
        raise RuntimeError(f"GROQ model call failed: {e}")

    try:
        raw = resp.choices[0].message.content.strip()
    except Exception:
        raw = str(resp)

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"AI returned no JSON: {raw[:300]}")

    json_text = raw[start:end+1].strip()

    # Attempt multiple fixes
    try:
        plan = json.loads(json_text)
        if isinstance(plan, dict):
            return plan
    except Exception:
        pass

    try:
        txt2 = re.sub(r"'", '"', json_text)
        plan = json.loads(txt2)
        if isinstance(plan, dict):
            return plan
    except Exception:
        pass

    try:
        txt3 = re.sub(r",\s*}", "}", json_text)
        txt3 = re.sub(r",\s*]", "]", txt3)
        plan = json.loads(txt3)
        if isinstance(plan, dict):
            return plan
    except Exception:
        pass

    raise ValueError(f"AI returned JSON but parsing failed. Raw start: {raw[:400]}")

###############################################
# PROMPT BUILDER
###############################################
def build_prompt(question, schema, metadata, metrics):
    rules = (
        "- Financial Year rule: FY starts in March and ends in February next year.\n"
        "- When user mentions 'FY' or a plain 4-digit year, interpret as FinancialYear by default.\n"
        "- Use FinancialMonth/FinancialQuarter/FinancialYear columns when possible.\n"
        "- If user explicitly says 'transaction year/month', use TransactionYear/TransactionMonth.\n"
        "- Output STRICT valid JSON only. Do NOT include any explanation or extra text.\n"
        "- JSON keys must be: select (list), filters (list), group_by (list), order_by (list), limit (nullable).\n"
        "- Each select item: {\"column\": <column-name|null>, \"expression\": <sql-expr|null>, \"aggregation\": <SUM|AVG|COUNT|MIN|MAX|null>, \"alias\": <alias>}.\n"
    )
    return f"""
You are a SQL semantic engine. Convert the user's question into STRICT JSON query plan (no explanation).

{rules}

USER QUESTION:
{question}

TABLE: {TABLE_NAME}
SCHEMA: {json.dumps(schema or {}, indent=2)}
BUSINESS METRICS: {json.dumps(metrics, indent=2)}

OUTPUT ONLY valid JSON.
"""

###############################################
# PLAN NORMALIZATION & METRIC FALLBACK
###############################################
AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None

def normalize_plan(plan: dict):
    plan = dict(plan or {})
    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    schema = get_schema() or {}

    # normalize group_by to list of strings
    gb = []
    for g in group_by:
        if isinstance(g, dict):
            c = g.get("column")
            if isinstance(c, str):
                gb.append(c)
        elif isinstance(g, str):
            gb.append(g)
    # keep only valid schema cols if schema present
    if schema:
        gb = [c for c in gb if c in schema]
    # preserve order but unique
    seen = set()
    gb_final = []
    for c in gb:
        if c not in seen:
            seen.add(c)
            gb_final.append(c)

    # normalize selects: ensure dicts with alias
    clean_selects = []
    seen_aliases = set()
    # ensure group_by columns are in selects
    for g in gb_final:
        clean_selects.append({"column": g, "expression": None, "aggregation": None, "alias": g})
        seen_aliases.add(g)

    for s in selects:
        if not isinstance(s, dict):
            continue
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or (col if col else "value")
        alias = alias.replace("%", "_pct") if isinstance(alias, str) else alias

        if isinstance(expr, str) and is_aggregate_expression(expr):
            agg = None
        if agg is not None and (str(agg).lower() in ("none","null")):
            agg = None

        if alias not in seen_aliases:
            clean_selects.append({"column": col, "expression": expr, "aggregation": agg, "alias": alias})
            seen_aliases.add(alias)

    # normalize order_by: keep only items referencing select aliases or group_by
    sel_aliases = {s["alias"] for s in clean_selects if s.get("alias")}
    cleaned_order = []
    for o in order_by:
        if not isinstance(o, dict):
            continue
        col = o.get("column")
        direction = o.get("direction", "DESC") or "DESC"
        if col and (col in sel_aliases or col in gb_final):
            cleaned_order.append({"column": col, "direction": direction})

    plan["select"] = clean_selects
    plan["group_by"] = gb_final
    plan["order_by"] = cleaned_order
    plan["limit"] = limit
    plan["filters"] = plan.get("filters", []) or []

    return plan

def detect_metric_from_text(text: str):
    text_low = (text or "").lower()
    for syn, metric in SYNONYM_MAP.items():
        if re.search(r'\b' + re.escape(syn) + r'\b', text_low):
            return metric
    for syn, metric in SYNONYM_MAP.items():
        if syn in text_low:
            return metric
    return None

def build_plan_from_metric(metric_key, question_text):
    m = METRICS.get(metric_key)
    if not m:
        return None
    expr = m.get("expression")
    agg = m.get("aggregation")
    alias = metric_key

    if isinstance(expr, str) and re.search(r'^\s*(count|sum|avg|min|max)\s*\(', expr.strip(), re.IGNORECASE):
        select_item = {"column": None, "expression": expr, "aggregation": None, "alias": alias}
    else:
        if agg:
            select_item = {"column": None, "expression": expr, "aggregation": agg, "alias": alias}
        else:
            select_item = {"column": None, "expression": expr, "aggregation": None, "alias": alias}

    plan = {"select": [select_item], "filters": [], "group_by": [], "order_by": [], "limit": None}
    tf = parse_time_filters(question_text)
    for t in tf:
        t["value"] = sanitize_filter_value(t.get("value"))
    plan["filters"].extend(tf)
    return plan

###############################################
# VALUE FILTER EXTRACTION (improved)
###############################################
def extract_value_filters(question: str, schema: dict):
    """
    Best-effort extraction of value filters. Tries:
    1) direct "column value" patterns (case-insensitive)
       e.g. "transportmode sea" or "Transport Mode = sea" or "mode sea"
    2) synonyms/dimension-based extraction (e.g., "by customer X")
    3) avoid producing one filter per column (use only detected columns)
    """
    q = (question or "").lower()
    filters = []

    if not schema:
        return []

    # build simplified schema name map for fuzzy matching
    simplified_cols = {col.lower(): col for col in schema.keys()}

    # common separators and words that often precede values
    pattern_after = r"(?:is|=|:|=|for|of|=)?\s*['\"]?([A-Za-z0-9\-\_& ]{1,60})['\"]?"

    # 1) direct column mentions followed by value
    for col_low, col_exact in simplified_cols.items():
        # try exact column mention
        # allow spaces/different casing: match words of column name pieces in text
        col_tokens = re.sub(r"[_\s]+", r"\\s*", re.escape(col_low))
        m = re.search(rf"\b{col_tokens}\b\s*{pattern_after}", q)
        if m:
            val = m.group(1).strip()
            # skip plain year matches (time filter handles years)
            if re.match(r"20\d{2}$", val):
                continue
            filters.append({"column": col_exact, "operator": "=", "value": val})
            continue

    # 2) dimension synonyms like "by customer X" or "for customer X"
    for syn, mapped_col in DIM_LOOKUP.items():
        if syn in q:
            # look for "syn <value>" patterns
            m = re.search(rf"\b{re.escape(syn)}\b\s*(?:is|=|:|for|of|by)?\s*['\"]?([A-Za-z0-9\-\_& ]{{1,60}})['\"]?", q)
            if m:
                val = m.group(1).strip()
                if mapped_col in schema and not re.match(r"20\d{2}$", val):
                    filters.append({"column": mapped_col, "operator": "=", "value": val})
                    # prefer synonym-based extraction: do not add other values for same column
                    continue

    # 3) last-resort: look for small set of candidate columns often filtered by values (common dimensions)
    candidate_cols = ["TransportMode", "TransportModeCode", "JobType", "JobStatus", "CompanyCode", "BranchCode", "CustomerLeadGroup", "OriginCountry", "DestinationCountry"]
    for col in candidate_cols:
        if col in schema and col.lower() in q:
            # already handled by direct match above;
            continue
        # attempt to detect patterns like "sea", "air", "road"
        if re.search(r"\b(sea|air|road|rail|truck)\b", q) and "Transport" in col:
            # pick the actual word
            m = re.search(r"\b(sea|air|road|rail|truck)\b", q)
            if m:
                filters.append({"column": col, "operator": "=", "value": m.group(1)})
                break

    # deduplicate: keep first filter per column
    dedup = {}
    for f in filters:
        c = f.get("column")
        if c not in dedup:
            dedup[c] = f
    return list(dedup.values())

###############################################
# TOP / DIRECTION / DIMENSION DETECTION
###############################################
DESC_KEYWORDS = {"top","highest","higher","greater","greatest","max","maximum","most","biggest"}
ASC_KEYWORDS = {"least","bottom","lowest","lower","minimum","min","smallest","fewest"}

def extract_top_n_and_direction(text: str):
    t = (text or "").lower()
    # top N
    m = re.search(r'\btop\s+(\d+)\b', t)
    n = int(m.group(1)) if m else None
    # direction
    dir_detect = None
    for w in DESC_KEYWORDS:
        if re.search(r'\b' + re.escape(w) + r'\b', t):
            dir_detect = "DESC"
            break
    for w in ASC_KEYWORDS:
        if re.search(r'\b' + re.escape(w) + r'\b', t):
            dir_detect = "ASC"
            break
    has_top = bool(re.search(r'\btop\b', t))
    return n, dir_detect, has_top

def detect_dimension_from_text(text: str, schema: dict):
    t = (text or "").lower()
    if schema:
        for col in schema.keys():
            if col.lower() in t:
                return col
    for syn, col in DIM_LOOKUP.items():
        if syn in t and (not schema or col in schema):
            return col
    # fallback: look for "by X" patterns and map X using DIM_LOOKUP
    m = re.search(r'\bby\s+([a-z ]{2,30})\b', t)
    if m:
        w = m.group(1).strip()
        if w in DIM_LOOKUP and DIM_LOOKUP[w] in schema:
            return DIM_LOOKUP[w]
    return None

###############################################
# MAIN: extract_query()
###############################################
def extract_query(question: str):
    schema = get_schema() or {}
    client = get_client()

    prompt = build_prompt(question, schema, METADATA, METRICS)
    model = choose_best_groq_model(client) if client else None

    # call model (best-effort)
    plan = None
    if client and model:
        try:
            plan = call_model_and_get_plan(client, model, prompt)
        except Exception as e:
            # don't fail hard; fallback later
            print("⚠ model parse error:", e)
            plan = None

    # fallback to metric-only plan
    if not plan:
        metric_key = detect_metric_from_text(question)
        if metric_key:
            plan = build_plan_from_metric(metric_key, question)
        else:
            return {"error": "Could not parse question"}

    if not isinstance(plan, dict):
        return {"error": "Plan returned is not a dict", "raw": plan}

    # authoritative time filters
    time_filters = parse_time_filters(question)
    for tf in time_filters:
        tf["value"] = sanitize_filter_value(tf.get("value"))

    # extracted value filters from question
    user_value_filters = extract_value_filters(question, schema)

    # merge filters:
    final_filters = []
    # start with authoritative time filters
    final_filters.extend(time_filters)
    # add model filters that are not time columns
    model_filters = plan.get("filters", []) or []
    time_cols = {t.get("column") for t in time_filters}
    for f in model_filters:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        if col not in time_cols:
            f["value"] = sanitize_filter_value(f.get("value"))
            final_filters.append(f)
    # add user-extracted value filters (do not override existing filters for same column)
    existing_cols = {f.get("column") for f in final_filters}
    for vf in user_value_filters:
        if vf.get("column") not in existing_cols:
            vf["value"] = sanitize_filter_value(vf.get("value"))
            final_filters.append(vf)

    plan["filters"] = final_filters

    # dimension/top/ordering logic
    dim_col = detect_dimension_from_text(question, schema)
    top_n, direction, has_top_word = extract_top_n_and_direction(question)

    selects = plan.get("select", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    # find metric alias
    metric_alias = None
    if selects:
        for s in selects:
            if isinstance(s, dict) and (s.get("aggregation") or s.get("expression")):
                metric_alias = s.get("alias") or s.get("column") or "value"
                break
        if not metric_alias and isinstance(selects[0], dict):
            metric_alias = selects[0].get("alias") or selects[0].get("column") or "value"

    # add dimension grouping if requested
    if dim_col and dim_col not in group_by:
        # prefer exact casing from schema
        if schema:
            for col in schema.keys():
                if col.lower() == dim_col.lower():
                    dim_col = col
                    break
        group_by.insert(0, dim_col)
        # ensure dimension appears in select
        if not any(isinstance(s, dict) and (s.get("column") == dim_col or s.get("alias") == dim_col) for s in selects):
            selects.insert(0, {"column": dim_col, "expression": None, "aggregation": None, "alias": dim_col})

    # apply top N
    if top_n and not limit:
        plan["limit"] = top_n
        limit = top_n

    # ordering
    if metric_alias:
        if direction:
            order_by = [{"column": metric_alias, "direction": direction}]
        elif dim_col and not order_by:
            order_by = [{"column": metric_alias, "direction": "DESC"}]

    # default limit for plain 'top' word
    if has_top_word and not plan.get("limit"):
        plan["limit"] = 10

    plan["select"] = selects
    plan["group_by"] = group_by
    plan["order_by"] = order_by

    # final normalization
    plan = normalize_plan(plan)
    return plan
