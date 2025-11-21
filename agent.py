# agent.py
from groq import Groq
import os, json, pyodbc, re
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"

# ----------------------------------------
# Categorical Columns
# ----------------------------------------
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

# ----------------------------------------
# Load SQL Schema
# ----------------------------------------
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
    except:
        return {c: "varchar" for c in CATEGORICAL_COLUMNS}

_SCHEMA = {}

def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA

# Numeric columns
def numeric_columns():
    schema = get_schema()
    return [
        c for c, t in schema.items()
        if t in ("decimal", "numeric", "money", "float", "int", "bigint", "smallint")
    ]

# ----------------------------------------
# Time Parsing
# ----------------------------------------
def parse_time_from_text(question):
    q = question.lower()
    now = datetime.utcnow()

    res = {"year": None, "quarter": None, "month": None, "timeframe": None}

    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        res["year"] = int(m.group(1))

    if "last year" in q or "previous year" in q:
        res["year"] = now.year - 1
        res["timeframe"] = "previous_year"

    if "last quarter" in q:
        current_q = (now.month - 1)//3 + 1
        prev_q = 4 if current_q == 1 else current_q - 1
        prev_year = now.year - 1 if current_q == 1 else now.year
        res["quarter"] = prev_q
        res["year"] = prev_year
        res["timeframe"] = "last_quarter"

    m = re.search(r'(?:q|quarter)\D*([1-4])(?:.*?(20\d{2}))?', q)
    if m:
        res["quarter"] = int(m.group(1))
        if m.group(2):
            res["year"] = int(m.group(2))

    if "last month" in q:
        prev = now.replace(day=1) - timedelta(days=1)
        res["month"] = prev.month
        res["year"] = prev.year
        res["timeframe"] = "last_month"

    months = {
        m.lower(): i for i, m in enumerate([
            "", "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ])
    }

    for name, idx in months.items():
        if name and name in q:
            res["month"] = idx
            if not res["year"]:
                res["year"] = now.year
            break

    return res

# ----------------------------------------
# Category & Metric Detection
# ----------------------------------------
def find_metric_from_text(q):
    schema = get_schema()
    q = q.lower()

    for col in numeric_columns():
        if col.lower() in q:
            return col

    if "profit" in q:
        return "JobProfit"
    if "revenue" in q or "sales" in q:
        return "REVAmount"

    nums = numeric_columns()
    return nums[0] if nums else None


def find_category_from_text(q):
    q = q.lower()
    schema = get_schema()

    for col in CATEGORICAL_COLUMNS:
        if col.lower() in q:
            return col

    if "transport" in q:
        return "TransportMode"

    return None


def extract_category_value(question, col):
    codes = re.findall(r'\b[A-Z]{2,6}\b', question)
    if codes:
        return codes[0]
    return None

# ----------------------------------------
# LLM Client
# ----------------------------------------
def get_client():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None

# ----------------------------------------
# Main Query Extractor
# ----------------------------------------
def extract_query(question):
    schema = get_schema()
    q_lower = question.lower()

    metric = find_metric_from_text(question)

    agg = "sum"
    if "avg" in q_lower:
        agg = "avg"
    if "count" in q_lower:
        agg = "count"

    time = parse_time_from_text(question)

    group_col = find_category_from_text(question)
    group_by = "by " in q_lower or "breakdown" in q_lower or "group" in q_lower

    category_value = extract_category_value(question, group_col) if group_col else None

    compare = []
    if "compare" in q_lower or "vs" in q_lower:
        compare = re.findall(r'\b[A-Z]{2,6}\b', question)

    # -------------------------
    # LLM safe override
    # -------------------------
    client = get_client()
    if client:
        try:
            prompt = {
                "question": question,
                "columns": list(schema.keys())
            }
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": json.dumps(prompt)}],
                temperature=0
            )

            parsed = json.loads(resp.choices[0].message.content)

            if parsed.get("metric"):
                metric = parsed["metric"]

            # *** SAFE TIME MERGE FIX ***
            llmtime = parsed.get("time", {})
            if time.get("year") is None and llmtime.get("year"):
                time["year"] = llmtime["year"]
            if llmtime.get("quarter"):
                time["quarter"] = llmtime["quarter"]
            if llmtime.get("month"):
                time["month"] = llmtime["month"]

            llg = parsed.get("group_column")
            if llg in schema:
                group_col = llg
            if parsed.get("group_by") is True:
                group_by = True

            if parsed.get("category_value"):
                category_value = parsed["category_value"]

            if parsed.get("compare"):
                compare = parsed["compare"]

        except Exception as e:
            print("LLM failed:", e)

    return {
        "metric": metric,
        "aggregation": agg,
        "time": time,
        "group_by": group_by,
        "group_column": group_col,
        "category_value": category_value,
        "compare": compare
    }
