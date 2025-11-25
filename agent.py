# agent.py — JSON-safe, Qwen-fixed, stable version
import os
import json
import re
from datetime import datetime, timedelta
from math import ceil
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"

def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

METADATA = load_json_file("metadata.json")
METRICS = load_json_file("metrics.json")

# -------------------------
# SQL Schema
# -------------------------
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
        return {}

_SCHEMA = {}

def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA

# -------------------------
# Groq client — QWEN ONLY
# -------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None

def choose_best_groq_model(client):
    # FORCE the stable model ONLY
    return "qwen/qwen3-32b"

# -------------------------
# Time parsing — unchanged
# -------------------------
def month_name_to_num(name):
    try:
        return datetime.strptime(name[:3].capitalize(), "%b").month
    except:
        return None

def calendar_to_fy_month(calendar_month):
    return ((calendar_month - 3) % 12) + 1

def calendar_to_fy_year(year, month):
    return year if month >= 3 else year - 1

def parse_time_filters(text):
    q = text.lower()
    filters = []

    m = re.search(r'\b(20\d{2})\b', q)
    if m:
        filters.append({"column": "FinancialYear", "operator": "=", "value": int(m.group(1))})

    return filters

# -------------------------
# LLM Prompt
# -------------------------
def build_prompt(question, schema, metadata, metrics):
    return f"""
Convert the question to JSON SQL plan. STRICT JSON ONLY.

User question:
{question}

TABLE: {TABLE_NAME}
SCHEMA: {json.dumps(schema)}
METRICS: {json.dumps(metrics)}

Output JSON keys: select, filters, group_by, order_by, limit
"""

# -------------------------
# Extract plan — SUPER SAFE
# -------------------------
def extract_query(question):
    client = get_client()
    if not client:
        return {"error": "Missing Groq API Key"}

    model = choose_best_groq_model(client)
    schema = get_schema()
    prompt = build_prompt(question, schema, METADATA, METRICS)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
    except Exception as e:
        return {"error": f"Groq error: {e}"}

    raw = response.choices[0].message.content.strip()

    # Extract JSON only
    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        return {"error": "invalid_json", "raw": raw}

    json_text = raw[start:end+1]

    try:
        plan = json.loads(json_text)
    except:
        return {"error": "json_parse_failed", "raw": raw, "json_text": json_text}

    # required keys check
    for k in ["select", "filters", "group_by", "order_by", "limit"]:
        if k not in plan:
            return {"error": "missing_keys", "raw": raw, "plan": plan}

    return plan
