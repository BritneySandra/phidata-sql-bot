# agent.py — Fully Dynamic LLM SQL Agent (Updated Version)
# ---------------------------------------------------------
# Fixes added:
# 1. JSON-only extraction (ignores LLM explanations)
# 2. Percent (%) sanitizer to avoid pyodbc "Invalid format specifier"
# 3. Robust JSON parsing
# ---------------------------------------------------------

import os
import json
import pyodbc
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"

# ------------------------------------------------------
# Load SQL Schema
# ------------------------------------------------------
def load_sql_schema():
    try:
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
    except Exception as e:
        print("❌ Failed to load SQL schema:", e)
        return {}

_SCHEMA = {}

def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = load_sql_schema()
    return _SCHEMA


# ------------------------------------------------------
# Load metadata.json
# ------------------------------------------------------
def load_metadata():
    try:
        with open("metadata.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print("❌ Cannot load metadata.json:", e)
        return {}

METADATA = load_metadata()


# ------------------------------------------------------
# Load metrics.json
# ------------------------------------------------------
def load_metrics():
    try:
        with open("metrics.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print("❌ Cannot load metrics.json:", e)
        return {}

METRICS = load_metrics()


# ------------------------------------------------------
# LLM client
# ------------------------------------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


# ------------------------------------------------------
# Build prompt for LLM
# ------------------------------------------------------
def build_prompt(question, schema, metadata, metrics):

    return f"""
You are an expert SQL Semantic Engine.

Convert the user question into a STRICT JSON query plan.

USER QUESTION:
{question}

SQL TABLE:
{TABLE_NAME}

SCHEMA:
{json.dumps(schema, indent=2)}

COLUMN DESCRIPTIONS:
{json.dumps(metadata, indent=2)}

BUSINESS METRICS:
{json.dumps(metrics, indent=2)}

OUTPUT STRICT JSON ONLY. NO TEXT.
FORMAT:
{{
  "select": [
    {{
      "column": "column_name_or_null",
      "expression": "sql_expression_or_null",
      "aggregation": "SUM|AVG|COUNT|etc",
      "alias": "metric_name"
    }}
  ],
  "filters": [
    {{
      "column": "ColumnName",
      "operator": "=",
      "value": any
    }}
  ],
  "group_by": ["ColumnName"],
  "order_by": [
    {{
      "column": "AliasOrColumn",
      "direction": "ASC|DESC"
    }}
  ],
  "limit": number_or_null
}}
"""


# ------------------------------------------------------
# Extract Query via LLM
# ------------------------------------------------------
def extract_query(question: str):
    schema = get_schema()
    client = get_client()

    if not client:
        return {"error": "Missing GROQ_API_KEY"}

    # Send prompt
    prompt = build_prompt(question, schema, METADATA, METRICS)

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message["content"].strip()

    # ----------------------------
    # STEP 1: Only keep JSON {...}
    # ----------------------------
    json_start = raw.find("{")
    json_end = raw.rfind("}")

    if json_start == -1 or json_end == -1:
        return {
            "error": "LLM did not return JSON.",
            "raw_response": raw
        }

    json_text = raw[json_start:json_end + 1]

    # ----------------------------------------------------
    # STEP 2: FIX — Remove '%' to avoid SQL format errors
    # ----------------------------------------------------
    json_text = json_text.replace("%", "_pct")

    # ----------------------------------------------
    # STEP 3: Attempt JSON parse
    # ----------------------------------------------
    try:
        plan = json.loads(json_text)
        return plan

    except Exception as e:
        return {
            "error": f"Failed to parse LLM JSON: {e}",
            "json_received": json_text,
            "raw_response": raw
        }
