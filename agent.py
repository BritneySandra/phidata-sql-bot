# agent.py — Fully Dynamic Auto-Model Groq SQL Agent
# --------------------------------------------------------------------
# Features:
# ✔ Auto-detect strongest available Groq model (70B > 32B > 8B)
# ✔ No model decommission errors
# ✔ JSON-only extraction
# ✔ % sanitizer
# ✔ Fully dynamic SQL intent extraction
# ✔ Works with metadata.json + metrics.json
# --------------------------------------------------------------------

import os
import json
import pyodbc
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"


# ====================================================================
#  LOAD SQL SCHEMA
# ====================================================================
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


# ====================================================================
#  LOAD metadata.json
# ====================================================================
def load_metadata():
    try:
        with open("metadata.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print("❌ Cannot load metadata.json:", e)
        return {}

METADATA = load_metadata()


# ====================================================================
#  LOAD metrics.json
# ====================================================================
def load_metrics():
    try:
        with open("metrics.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print("❌ Cannot load metrics.json:", e)
        return {}

METRICS = load_metrics()


# ====================================================================
#  AUTO-DETECT BEST GROQ MODEL
# ====================================================================
def choose_best_groq_model(client):
    """
    Auto-select the strongest available model from Groq.
    Priority:
       1. llama-3.3-70b-versatile
       2. meta-llama/llama-4-scout-17b-16e-instruct
       3. qwen/qwen3-32b
       4. llama-3.1-8b-instant
    """
    preferred_order = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "llama-3.1-8b-instant"
    ]

    try:
        models = client.models.list()
        available = [m.id for m in models.data]

        for m in preferred_order:
            if m in available:
                print("🚀 Using Groq model:", m)
                return m

        # fallback to ANY LLM
        for m in available:
            if "llama" in m or "qwen" in m:
                print("⚠️ Using fallback model:", m)
                return m

        # last fallback
        print("⚠️ Using final fallback model:", available[0])
        return available[0]

    except Exception as e:
        print("❌ Could not fetch model list:", e)
        return "llama-3.1-8b-instant"


# ====================================================================
#  LLM CLIENT
# ====================================================================
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


# ====================================================================
#  PROMPT BUILDER
# ====================================================================
def build_prompt(question, schema, metadata, metrics):

    return f"""
You are an expert SQL reasoning engine.

Convert the user question into a STRICT JSON query plan only.

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

Return STRICT JSON with this structure:
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

NO extra text. NO explanations. JSON ONLY.
"""


# ====================================================================
#  MAIN FUNCTION — EXTRACT QUERY
# ====================================================================
def extract_query(question: str):
    schema = get_schema()
    client = get_client()

    if not client:
        return {"error": "Missing GROQ_API_KEY"}

    # Choose best available model
    model_name = choose_best_groq_model(client)

    prompt = build_prompt(question, schema, METADATA, METRICS)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message["content"].strip()

    # ---------------------------
    # Extract only JSON {...}
    # ---------------------------
    json_start = raw.find("{")
    json_end = raw.rfind("}")

    if json_start == -1 or json_end == -1:
        return {
            "error": "LLM did not return JSON",
            "raw_response": raw
        }

    json_text = raw[json_start:json_end + 1]

    # ---------------------------
    # Sanitize % symbols
    # ---------------------------
    json_text = json_text.replace("%", "_pct")

    # ---------------------------
    # Parse JSON
    # ---------------------------
    try:
        return json.loads(json_text)
    except Exception as e:
        return {
            "error": f"Failed to parse JSON: {e}",
            "json_received": json_text,
            "raw_response": raw
        }
