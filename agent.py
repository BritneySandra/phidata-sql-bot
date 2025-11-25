# agent.py — Fully Dynamic LLM SQL Agent
# -----------------------------------------------------
# This version removes all manual hardcoded logic.
# The LLM uses schema + metadata.json + metrics.json
# to dynamically infer filters, grouping, metrics,
# top N, comparisons, trends, periods, etc.
# -----------------------------------------------------

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
# Load metadata.json (column descriptions)
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
# Load metrics.json (business rules)
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
# LLM Client
# ------------------------------------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


# ------------------------------------------------------
# Build AI Prompt
# ------------------------------------------------------
def build_prompt(question, schema, metadata, metrics):

    return f"""
You are an advanced SQL Semantic Reasoning Engine.
Your job: Convert the user's question INTO A SQL QUERY PLAN IN JSON FORMAT.

Your inputs:
1. SQL schema of table {TABLE_NAME}
2. Column descriptions from metadata.json
3. Business metric formulas from metrics.json
4. The natural language user question

-----------------------------------
USER QUESTION:
{question}
-----------------------------------

SQL SCHEMA:
{json.dumps(schema, indent=2)}

COLUMN DESCRIPTIONS:
{json.dumps(metadata, indent=2)}

BUSINESS METRICS:
{json.dumps(metrics, indent=2)}

-----------------------------------
RULES FOR QUERY PLAN GENERATION:
-----------------------------------

✔ Determine *what metric* user wants.
✔ Infer metric using synonyms from metrics.json.
✔ Convert expressions exactly as written in metrics.json.
✔ Always reference columns using [ColumnName] format.

✔ Detect:
  • Filters (dimension, time, numeric ranges)
  • Group By (categorical columns)
  • Top N logic
  • Sorting (highest, lowest, larger, greater, smaller, least, top)
  • Trends (year-wise, month-wise)
  • Periods (last year, last month, previous quarter)
  • Comparisons (2023 vs 2024, revenue vs profit)
  • Derived metrics like profit %, job count, delays, etc.

✔ Use only REAL column names present in the schema.

✔ JSON OUTPUT FORMAT (STRICT):
{
  "select": [
    {
      "column": "column_name_or_null",
      "expression": "SQL expression or null",
      "aggregation": "SUM|AVG|COUNT|etc",
      "alias": "metric_name"
    }
  ],
  "filters": [
    {"column": "ColumnName", "operator": "=", "value": something}
  ],
  "group_by": ["ColumnName"],
  "order_by": [
    {"column": "AliasOrColumn", "direction": "ASC|DESC"}
  ],
  "limit": integer_or_null
}

Return ONLY JSON. NO text explanation.
"""


# ------------------------------------------------------
# Extract Query via LLM
# ------------------------------------------------------
def extract_query(question: str):
    schema = get_schema()
    client = get_client()

    if not client:
        return {"error": "Missing GROQ_API_KEY"}

    prompt = build_prompt(question, schema, METADATA, METRICS)

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message["content"]

    try:
        plan = json.loads(raw)
        return plan

    except Exception as e:
        # Return raw response so debugging is easy
        return {
            "error": f"Failed to parse LLM JSON: {e}",
            "raw_response": raw
        }
