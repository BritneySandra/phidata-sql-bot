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
        return {}

_SCHEMA = {}

def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA

# -------------------------
# classify columns
# -------------------------
def categorical_columns():
    schema = get_schema()
    cats = [c for c,t in schema.items() if t in ('varchar','nvarchar','char','text')]
    # include some codes too if present
    for code_col in ['CompanyCode','BranchCode','DeptCode','CustomerCode','LocalClientCode']:
        if code_col in schema and code_col not in cats:
            cats.append(code_col)
    return cats

def numeric_columns():
    schema = get_schema()
    nums = [c for c,t in schema.items() if t in ('decimal','numeric','money','float','int','bigint','smallint')]
    return nums

# -------------------------
# metric synonyms & category synonyms
# -------------------------
METRIC_SYNONYMS = {
    "revenue": ["rev","revenue","sales","turnover","income"],
    "REVAmount": ["revamount","revenue","sales"],
    "JobProfit": ["profit","jobprofit","margin","earnings"],
    "CSTAmount": ["cost","expense","cst","costs"],
    "ACRAmount": ["acr","accrual"],
    # fallback measures many exist in schema - we prefer REVAmount and JobProfit
}

# map common category words to actual column names (expand as needed)
CATEGORY_SYNONYMS = {
    "transport": ["transport","mode","transportmode","transmode"],
    "transportmode": ["transportmode","mode"],
    "container": ["container","containermode","container mode"],
    "product": ["product","productlevel","product level","productlevel1","productlevel2","productlevel3"],
    "department": ["department","dept","deptcode"],
    "branch": ["branch","branchcode"],
    "customer": ["customer","customername","customer code","customer code"],
    "company": ["company","companycode"],
    "jobtype": ["job type","jobtype"],
    "joblevel1": ["joblevel1","job level1"],
    "joblevel2": ["joblevel2","job level2"],
    "joblevel3": ["joblevel3","job level3"],
    "country": ["country","countryname","origincountry","destinationcountry"],
    "year": ["year","financialyear","fy","fyyear"],
    "quarter": ["quarter","financialquarter","fyquarter"]
}

# helper: best metric from question
def find_metric_from_text(question):
    q = question.lower()
    # check explicit metric column mention
    schema = get_schema()
    for col in numeric_columns():
        if col.lower() in q:
            return col
    # synonyms mapping
    for metric_col, syns in METRIC_SYNONYMS.items():
        for s in syns:
            if s in q:
                # if metric_col is a friendly key like "revenue" map to REVAmount if exists
                if metric_col in schema:
                    return metric_col
                # else if REVAmount exists, prefer it
                if 'REVAmount' in schema and 'revenue' in syns:
                    return 'REVAmount'
                if 'JobProfit' in schema and 'profit' in syns:
                    return 'JobProfit'
    # fallback heuristics
    if 'profit' in q and 'JobProfit' in schema:
        return 'JobProfit'
    if 'revenue' in q and 'REVAmount' in schema:
        return 'REVAmount'
    # final fallback - choose a primary measure if present
    schema_nums = numeric_columns()
    for preferred in ['REVAmount','USDREVAmount','JobProfit','REVAmount']:
        if preferred in schema_nums:
            return preferred
    return schema_nums[0] if schema_nums else None

# helper: find category column from question using synonyms and schema
def find_category_from_text(question):
    q = question.lower()
    schema = get_schema()
    # try direct match with column names
    for col in categorical_columns():
        if col.lower() in q:
            return col
    # synonyms map
    for canonical, syns in CATEGORY_SYNONYMS.items():
        for s in syns:
            if s in q:
                # try to find actual column in schema that matches canonical
                # common mapping:
                mapping = {
                    "transportmode": ["TransportMode"],
                    "container": ["ContainerMode","Container"],
                    "product": ["ProductLevel1","ProductLevel2","ProductLevel3"],
                    "department": ["DeptCode","DeptPK"],
                    "branch": ["BranchCode","BranchPK"],
                    "customer": ["CustomerName","CustomerCode","LocalClientName"],
                    "company": ["CompanyCode","CompanyPK"],
                    "jobtype": ["JobType"],
                    "joblevel1": ["JobLevel1"],
                    "joblevel2": ["JobLevel2"],
                    "joblevel3": ["JobLevel3"],
                    "country": ["CountryName","OriginCountry","DestinationCountry"]
                }
                candidates = mapping.get(canonical, [])
                for c in candidates:
                    if c in schema:
                        return c
    # fallback: if user asked "by" but no direct mapping, return TransportMode if present
    if 'by ' in q or 'breakdown' in q or 'group' in q or 'split' in q:
        if 'TransportMode' in schema:
            return 'TransportMode'
    return None

# -------------------------
# time parsing helper
# -------------------------
def parse_time_from_text(question):
    """Return dict with keys: year (int or None), quarter (int or None), month (int or None)"""
    q = question.lower()
    now = datetime.utcnow()
    res = {"year": None, "quarter": None, "month": None, "timeframe": None}
    # explicit year: "in 2024" or "2024"
    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        res["year"] = int(m.group(1))
    # last year / previous year
    if 'previous year' in q or 'last year' in q:
        res["year"] = now.year - 1
        res["timeframe"] = "previous_year"
    # quarter patterns
    qmatch = re.search(r'last quarter', q)
    if qmatch:
        # determine current quarter and subtract 1
        current_q = (now.month - 1)//3 + 1
        prev_q = current_q - 1
        prev_year = now.year
        if prev_q == 0:
            prev_q = 4
            prev_year -= 1
        res["quarter"] = prev_q
        res["year"] = prev_year
        res["timeframe"] = "last_quarter"
    # explicit quarter like "Q1 2024" or "quarter 3 2024"
    m = re.search(r'(?:q|quarter)[^\d]*([1-4])(?:[^0-9]+(20\d{2}))?', q)
    if m:
        res["quarter"] = int(m.group(1))
        if m.group(2):
            res["year"] = int(m.group(2))
    # last month, previous month
    if 'last month' in q or 'previous month' in q:
        prev = now.replace(day=1) - timedelta(days=1)
        res["month"] = prev.month
        res["year"] = prev.year
        res["timeframe"] = "last_month"
    # month name like "march 2024" or "mar 2024"
    months = {m.lower(): i for i,m in enumerate(["","january","february","march","april","may","june","july","august","september","october","november","december"])}
    for name, idx in months.items():
        if name and name in q:
            res["month"] = idx
            # if year already present keep it; else maybe current year
            if not res["year"]:
                res["year"] = now.year
            break
    return res

# -------------------------
# Build parsed object
# -------------------------
def extract_query(question):
    """
    Returns a dict with:
      metric: column name,
      aggregation: sum|avg|count|max|min
      time: dict(year, quarter, month)
      group_by: bool
      group_column: column name or None
      compare: list of values to compare (e.g., ['SEA','AIR'])
    """
    schema = get_schema()
    if not schema:
        # fallback minimal behavior without schema
        metric = find_metric_from_text(question)
        return {
            "metric": metric,
            "aggregation": "sum",
            "time": parse_time_from_text(question),
            "group_by": False,
            "group_column": None,
            "compare": []
        }

    # initial heuristics
    q_lower = question.lower()
    metric = find_metric_from_text(question)
    if metric is None:
        # pick default
        metric = 'REVAmount' if 'REVAmount' in schema else (numeric_columns()[0] if numeric_columns() else None)

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

    # detect group-by intent
    group_by = False
    group_col = None
    if any(k in q_lower for k in [" by ", " breakdown", " group ", " split ", " per ", " distribution", "compare", "compare "]):
        # try to find category in text
        group_col = find_category_from_text(question)
        if group_col:
            group_by = True
        else:
            # if phrase "by <word>" attempt to extract the word after 'by'
            m = re.search(r'by\s+([a-z0-9 _-]{2,40})', q_lower)
            if m:
                candidate = m.group(1).strip()
                # match candidate against category synonyms and column names
                best = find_category_from_text(candidate)
                if best:
                    group_col = best
                    group_by = True
                else:
                    # fallback to TransportMode if available
                    if 'TransportMode' in schema:
                        group_col = 'TransportMode'
                        group_by = True

    # detect compare values (e.g., "compare sea and air")
    compare = []
    if 'compare' in q_lower or 'vs ' in q_lower or ' vs ' in q_lower:
        # try to extract known category values from question for group_col
        possible_values = re.findall(r'\b[A-Z]{2,6}\b', question)  # e.g., SEA AIR COU
        # also extract words after 'compare' or separated by 'and'
        if not possible_values:
            parts = re.split(r'compare|vs|and|vs\.|vs,', q_lower)
            if len(parts) > 1:
                cand = parts[1]
                tokens = re.findall(r'\b[a-zA-Z]{2,20}\b', cand)
                compare = [t.strip().upper() for t in tokens][:4]
        else:
            compare = [v for v in possible_values]
    # time parse
    time = parse_time_from_text(question)

    return {
        "metric": metric,
        "aggregation": agg,
        "time": time,
        "group_by": group_by,
        "group_column": group_col,
        "compare": compare
    }
