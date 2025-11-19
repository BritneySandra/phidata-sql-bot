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

# -------------------------
# Explicit categorical list (your confirmed list)
# -------------------------
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
        # fallback: build from our known categorical columns (best-effort)
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
    nums = [c for c,t in schema.items() if t in ('decimal','numeric','money','float','int','bigint','smallint')]
    return nums

# -------------------------
# Metric synonyms
# -------------------------
METRIC_SYNONYMS = {
    "revenue": ["revenue", "rev", "sales", "turnover", "income"],
    "REVAmount": ["revamount", "revamount", "rev_amount"],
    "jobprofit": ["profit", "jobprofit", "margin"],
    "JobProfit": ["jobprofit", "profit"],
    "cost": ["cost", "cst", "expense", "cstamount"],
    "CSTAmount": ["cstamount", "cost"]
}

# category synonyms -> mapping to preferred column(s)
CATEGORY_SYNONYMS = {
    "transport": ["TransportMode"],
    "transportmode": ["TransportMode"],
    "container": ["ContainerMode"],
    "product": ["ProductLevel1","ProductLevel2","ProductLevel3"],
    "department": ["DeptCode"],
    "branch": ["BranchCode","AdjustedBranchCode"],
    "customer": ["CustomerName","CustomerCode"],
    "company": ["CompanyCode"],
    "jobtype": ["JobType"],
    "joblevel1": ["JobLevel1"],
    "joblevel2": ["JobLevel2"],
    "joblevel3": ["JobLevel3"],
    "country": ["CountryName","OriginCountry","DestinationCountry"]
}

TRANSPORT_KEYWORDS = ["SEA","AIR","ROA","COU","FSA","NOJ","UNKNOWN"]

# -------------------------
# Time parsing helper
# -------------------------
def parse_time_from_text(question):
    q = question.lower()
    now = datetime.utcnow()
    res = {"year": None, "quarter": None, "month": None, "timeframe": None}
    # explicit year
    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        res["year"] = int(m.group(1))
    # last year / previous year
    if 'previous year' in q or 'last year' in q:
        res["year"] = now.year - 1
        res["timeframe"] = "previous_year"
    # last quarter
    if 'last quarter' in q or 'previous quarter' in q:
        current_q = (now.month - 1)//3 + 1
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
    # month name
    months = {m.lower(): i for i,m in enumerate(["","january","february","march","april","may","june","july","august","september","october","november","december"])}
    for name, idx in months.items():
        if name and name in q:
            res["month"] = idx
            if not res["year"]:
                res["year"] = now.year
            break
    return res

# -------------------------
# Helpers to detect metric & category
# -------------------------
def find_metric_from_text(question):
    q = question.lower()
    schema = get_schema()
    # explicit mention of numeric column
    for col in numeric_columns():
        if col.lower() in q:
            return col
    # synonyms mapping
    for canonical, syns in METRIC_SYNONYMS.items():
        for s in syns:
            if s in q:
                # prefer canonical if in schema else best-match numeric
                if canonical in schema:
                    return canonical
    # heuristics
    if 'profit' in q:
        if 'JobProfit' in schema:
            return 'JobProfit'
    if 'revenue' in q:
        if 'REVAmount' in schema:
            return 'REVAmount'
    # fallback to first numeric column
    nums = numeric_columns()
    return nums[0] if nums else None

def find_category_from_text(question):
    q = question.lower()
    schema = get_schema()
    # direct column name mention
    for col in CATEGORICAL_COLUMNS:
        if col.lower() in q and col in schema:
            return col
    # synonyms mapping
    for key, targets in CATEGORY_SYNONYMS.items():
        for token in [key] + CATEGORY_SYNONYMS.get(key, []):
            if token in q:
                for t in targets:
                    if t in schema:
                        return t
    # phrase "by <word>" fallback
    m = re.search(r'by\s+([a-z0-9 _-]{2,40})', q)
    if m:
        candidate = m.group(1).strip()
        # try to map candidate to a column
        for col in CATEGORICAL_COLUMNS:
            if candidate.replace(" ", "").lower() in col.lower():
                if col in schema:
                    return col
    # default fallback to TransportMode if present
    if 'transport' in q and 'TransportMode' in schema:
        return 'TransportMode'
    return None

# ---------- category value detection ----------
def extract_category_value(question, category_col):
    """
    Attempts to extract a value for the detected category column.
    Returns string (value) or None.
    Heuristics:
     - look for patterns like "<category> <value>" or "<category> is <value>"
     - look for known short codes (SEA, AIR, COU etc.)
     - look for quoted phrases or capitalized multi-word tokens
    """
    q = question
    # 1) look for known code tokens (SEA, AIR, COU, FSA, NOJ, ROA)
    codes = re.findall(r'\b[A-Z]{2,6}\b', question)
    if codes:
        # ensure not matching year like 2023
        filtered = [c for c in codes if not re.match(r'\d{2,4}', c)]
        if filtered:
            return filtered[0]
    # 2) look for pattern "<category name> is X" or "transport mode sea"
    patterns = [
        rf'{category_col.lower()}[^\w0-9]+([A-Za-z0-9 &\-/]+)',
        rf'{category_col.lower().replace("_"," ")}[^\w0-9]+([A-Za-z0-9 &\-/]+)',
        r'for\s+([A-Za-z0-9 &\-/]+)\s+(?:in|on|for|$)',   # e.g., "profit for Air Export in 2023"
        r'for\s+the\s+([A-Za-z0-9 &\-/]+)\s',
        r'([A-Za-z0-9 &\-/]+)\s+transport',  # e.g., "Air Export transport"
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            val = m.group(1).strip(" .,:;\"'")
            # drop trailing words that are time tokens (year, quarter, last)
            val = re.sub(r'\b(20\d{2}|last|previous|quarter|q[1-4]|in)\b.*$','', val, flags=re.IGNORECASE).strip()
            if val:
                return val
    # 3) quoted phrase
    m = re.search(r'["\']([^"\']{2,60})["\']', q)
    if m:
        return m.group(1)
    return None

# -------------------------
# Groq client safe init
# -------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # allow working without LLM (fallback heuristics)
        return None
    return Groq(api_key=api_key)

# -------------------------
# Main extractor
# -------------------------
def extract_query(question):
    """
    Returns:
    {
      metric: str,
      aggregation: "sum"|"avg"|"count"|"max"|"min",
      time: {year, quarter, month, timeframe},
      group_by: bool,
      group_column: str or None,
      category_value: str or None,
      compare: [values],
    }
    """
    schema = get_schema()
    q_lower = question.lower()

    # metric detection
    metric = find_metric_from_text(question)

    # aggregation detection
    agg = "sum"
    if any(w in q_lower for w in ["average","avg","mean"]):
        agg = "avg"
    if any(w in q_lower for w in ["count","how many","number of"]):
        agg = "count"
    if any(w in q_lower for w in ["max","maximum","largest"]):
        agg = "max"
    if any(w in q_lower for w in ["min","minimum","smallest"]):
        agg = "min"

    # time
    time = parse_time_from_text(question)

    # group_by intent
    group_by = False
    group_col = find_category_from_text(question)
    if group_col:
        # if question contains words indicating grouping
        if any(k in q_lower for k in [" by ", "breakdown", "group", "split", "per ", "distribution", "compare", "vs", "versus"]):
            group_by = True

    # category value detection
    category_value = None
    if group_col:
        category_value = extract_category_value(question, group_col)
        # also check for single token after category key (e.g., "transport mode sea")
        if not category_value:
            m = re.search(rf'{group_col.lower()}(?:\s|:|=)?\s*([A-Za-z0-9 &\-/]+)', question, re.IGNORECASE)
            if m:
                v = m.group(1).strip()
                v = re.sub(r'\b(20\d{2}|in|on|for|last|previous|quarter|q[1-4])\b.*$','', v, flags=re.IGNORECASE).strip()
                if v:
                    category_value = v

    # compare extraction (e.g., compare sea and air)
    compare = []
    if 'compare' in q_lower or ' vs ' in q_lower or ' vs. ' in q_lower or ' versus ' in q_lower:
        # look for tokens separated by and / vs / , etc
        parts = re.split(r'compare|versus| vs | vs\.|,| and | & ', question, flags=re.IGNORECASE)
        # take likely tokens after compare
        for part in parts[1:]:
            tokens = re.findall(r'[A-Za-z0-9 &\-/]{1,40}', part)
            for t in tokens:
                t = t.strip()
                if t and not re.search(r'\b(last|previous|year|quarter|q[1-4])\b', t, re.IGNORECASE):
                    compare.append(t)
        # dedupe & uppercase short codes
        compare = [c for i,c in enumerate(compare) if c and c not in compare[:i]]

    # Try LLM if available (improves parsing) - **optional**
    client = get_client()
    if client:
        try:
            prompt = f"""
You are a JSON-only assistant. Return ONLY valid JSON.

Given SQL columns: {list(schema.keys())}

Convert the user question into JSON:
{{
  "metric": "<one numeric column or null>",
  "aggregation": "sum|avg|count|max|min",
  "time": {{"year": 2024, "quarter": null, "month": null}},
  "group_by": true|false,
  "group_column": "<one categorical column or null>",
  "category_value": "<string or null>",
  "compare": ["val1","val2"]
}}

User Question: "{question}"
Return JSON only.
"""
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=200
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            # merge / fallback values — prefer LLM where it's confident
            metric = parsed.get("metric") or metric
            agg = parsed.get("aggregation") or agg
            ltime = parsed.get("time") or {}
            time.update({k:v for k,v in ltime.items() if v is not None})
            # use LLM group_col if in schema
            lgroup = parsed.get("group_column")
            if lgroup and lgroup in schema:
                group_col = lgroup
                group_by = parsed.get("group_by", group_by)
            # category_value
            if parsed.get("category_value"):
                category_value = parsed.get("category_value")
            # compare
            if parsed.get("compare"):
                compare = parsed.get("compare")
        except Exception as e:
            # if LLM fails, continue with heuristics
            print("LLM parse skipped/fail:", e)

    # standardize compare cleanup
    if compare:
        compare = [c.strip() for c in compare if c and len(c.strip())>0]

    # normalize category_value to uppercase for common short codes like SEA/AIR
    if category_value:
        # if short uppercase code pattern found -> uppercase
        if re.fullmatch(r'[A-Za-z]{2,6}', category_value.strip()):
            category_value = category_value.strip().upper()
        else:
            category_value = category_value.strip()

    return {
        "metric": metric,
        "aggregation": agg,
        "time": time,
        "group_by": group_by,
        "group_column": group_col,
        "category_value": category_value,
        "compare": compare
    }
