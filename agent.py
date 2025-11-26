# agent.py — Fully AI-driven SQL Planner with Deterministic Corrections
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

# build synonym map
SYNONYM_MAP = {}
for k, v in METRICS.items():
    syns = v.get("synonyms", [])
    all_syns = [k.lower()] + [s.lower() for s in syns]
    for s in all_syns:
        SYNONYM_MAP[s] = k

############################################################
# DIMENSIONS
############################################################
DIMENSION_SYNONYMS = {
    "customer": ["customer", "client", "customerleadgroup", "customer group"],
    "branch": ["branch", "office", "location"],
    "company": ["company", "org"],
    "department": ["department", "dept", "team"],
    "jobtype": ["job type", "jobtype"],
    "transport": ["transport mode", "mode", "transport"],
}

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
            "Encrypt=no;TrustServerCertificate=yes;",
            timeout=5,
        )
        cur = conn.cursor()
        cur.execute(f"""
        SELECT COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME='{TABLE_NAME}'
        """)
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
# AI CLIENT
############################################################
def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except:
        return None

def choose_model(client):
    preferred = [
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "llama-3.1-8b-instant",
    ]
    try:
        available = [m.id for m in client.models.list().data]
        for p in preferred:
            if p in available:
                return p
        return available[0]
    except:
        return "llama-3.3-70b-versatile"


############################################################
# TIME INTELLIGENCE
############################################################
def calendar_to_fy_year(y, m): return y if m >= 3 else y - 1
def calendar_to_fy_month(m): return ((m - 3) % 12) + 1
def current(): return datetime.utcnow()

def previous_month():
    ref = current()
    first = ref.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.year, prev.month

def previous_year():
    return current().year - 1

def last_financial_quarter():
    ref = current()
    fy = calendar_to_fy_year(ref.year, ref.month)
    fm = calendar_to_fy_month(ref.month)
    fq = ceil(fm/3)
    if fq == 1:
        return fy - 1, 4
    return fy, fq - 1

def parse_time_filters(text: str):
    q = text.lower()
    filters = []

    # FY 2024
    m = re.search(r"\b(?:fy|financial year)\s*(20\d{2})\b", q)
    if m:
        filters.append({"column": "FinancialYear", "operator": "=", "value": int(m.group(1))})
        return filters

    # Quarter Q3 2024
    m = re.search(r"(?:q|quarter)\s*([1-4])(?:.*?(20\d{2}))?", q)
    if m:
        qn = int(m.group(1))
        yr = m.group(2)
        filters.append({"column": "FinancialQuarter", "operator": "=", "value": qn})
        if yr:
            filters.append({"column": "FinancialYear", "operator": "=", "value": int(yr)})
        return filters

    # Month + Year (Jan 2024)
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20\d{2})", q)
    if m:
        mon = datetime.strptime(m.group(1)[:3], "%b").month
        yr = int(m.group(2))
        filters.append({
            "column": "FinancialMonth",
            "operator": "=",
            "value": calendar_to_fy_month(mon),
        })
        filters.append({
            "column": "FinancialYear",
            "operator": "=",
            "value": calendar_to_fy_year(yr, mon),
        })
        return filters

    # previous month
    if "previous month" in q or "last month" in q:
        y, mth = previous_month()
        filters.append({"column": "FinancialMonth", "operator": "=", "value": calendar_to_fy_month(mth)})
        filters.append({"column": "FinancialYear", "operator": "=", "value": calendar_to_fy_year(y, mth)})
        return filters

    # previous quarter
    if "previous quarter" in q or "last quarter" in q:
        fy, fq = last_financial_quarter()
        filters.append({"column": "FinancialQuarter", "operator": "=", "value": fq})
        filters.append({"column": "FinancialYear", "operator": "=", "value": fy})
        return filters

    # previous year
    if "previous year" in q or "last year" in q:
        filters.append({"column": "FinancialYear", "operator": "=", "value": previous_year()})
        return filters

    # plain year
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        filters.append({"column": "FinancialYear", "operator": "=", "value": int(m.group(1))})

    return filters


############################################################
# METRIC DETECTION
############################################################
def detect_metric(text: str):
    t = text.lower()
    for syn, metric in SYNONYM_MAP.items():
        if syn in t:
            return metric
    return None

def build_metric_select(metric_key):
    m = METRICS.get(metric_key)
    expr = m.get("expression")
    agg = m.get("aggregation")

    if re.match(r"^\s*(sum|avg|min|max)\(", expr.strip(), re.IGNORECASE):
        return {"column": None, "expression": expr, "aggregation": None, "alias": metric_key}

    return {"column": None, "expression": expr, "aggregation": agg, "alias": metric_key}


############################################################
# TOP-N & ORDER
############################################################
DESC_WORDS = {"top", "highest", "biggest", "max", "most"}
ASC_WORDS = {"least", "lowest", "smallest", "min", "bottom"}

def detect_top_and_order(text):
    t = text.lower()
    m = re.search(r"top\s+(\d+)", t)
    n = int(m.group(1)) if m else None

    if any(w in t for w in DESC_WORDS): return n, "DESC"
    if any(w in t for w in ASC_WORDS): return n, "ASC"

    return n, None


############################################################
# DIMENSION DETECTION
############################################################
def detect_dimension(text: str, schema: dict):
    t = text.lower()

    for col in schema.keys():
        if col.lower() in t:
            return col

    for group, syns in DIMENSION_SYNONYMS.items():
        for s in syns:
            if s in t:
                for col in schema.keys():
                    if group.replace(" ", "") in col.lower() or group.lower() in col.lower():
                        return col

    return None


############################################################
# AI PROMPT BUILDER
############################################################
def prompt_builder(question, schema):
    return f"""
You are a senior SQL planning engine. Convert the question into a JSON SQL plan.

TABLE: {TABLE_NAME}

SCHEMA: {json.dumps(schema)}
METRICS: {json.dumps(METRICS)}

REQUIREMENTS:
- Output STRICT JSON ONLY.
- Keys allowed: select, filters, group_by, order_by, limit.
- SELECT must include alias for every expression.
- Use metric expressions from METRICS when referenced.
- Use FinancialYear, FinancialQuarter, FinancialMonth for dates.
- Never guess column names outside SCHEMA.
- If grouping is mentioned (e.g., 'by customer'), add GROUP BY.
- If user asks top/least N, add ORDER BY and LIMIT.
- If no aggregation needed, still produce valid expressions.
- DO NOT include comments.

QUESTION:
{question}
"""


############################################################
# AI CALL
############################################################
def call_ai(prompt):
    client = get_client()
    if not client:
        return None

    model = choose_model(client)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = resp.choices[0].message.content.strip()
    s = raw.find("{")
    e = raw.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(raw[s:e+1])
    except:
        return None


############################################################
# PLAN CORRECTION
############################################################
def fix_plan(plan, question, schema):
    if not isinstance(plan, dict):
        return plan

    plan.setdefault("select", [])
    plan.setdefault("filters", [])
    plan.setdefault("group_by", [])
    plan.setdefault("order_by", [])
    plan.setdefault("limit", None)

    # detect metric
    metric_key = detect_metric(question)
    if metric_key:
        metric_sel = build_metric_select(metric_key)
        if metric_sel not in plan["select"]:
            plan["select"].append(metric_sel)

    # add time filters
    tfs = parse_time_filters(question)
    for f in tfs:
        if f not in plan["filters"]:
            plan["filters"].append(f)

    # detect dimension
    dim = detect_dimension(question, schema)
    if dim:
        if dim not in plan["group_by"]:
            plan["group_by"].insert(0, dim)
        plan["select"].insert(0, {"column": dim, "expression": None, "aggregation": None, "alias": dim})

    # top & ordering
    n, direction = detect_top_and_order(question)
    if n:
        plan["limit"] = n

    # find metric alias
    metric_alias = None
    for s in plan["select"]:
        if s.get("alias") and (s.get("aggregation") or s.get("expression")):
            metric_alias = s["alias"]
            break

    if metric_alias:
        if direction:
            plan["order_by"] = [{"column": metric_alias, "direction": direction}]
        elif dim:
            plan["order_by"] = [{"column": metric_alias, "direction": "DESC"}]

    return plan


############################################################
# MAIN EXTRACTION FUNCTION
############################################################
def extract_query(question: str):
    schema = get_schema()
    prompt = prompt_builder(question, schema)
    ai_plan = call_ai(prompt)

    plan = ai_plan if ai_plan else {}

    plan = fix_plan(plan, question, schema)

    if not plan.get("select"):
        return {"error": "Cannot interpret the question"}

    return plan
