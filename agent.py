from groq import Groq
import os
import json
import pyodbc
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"


# --------------------------
# SQL COLUMN LOADING
# --------------------------
def load_sql_columns():
    """Load column names dynamically from SQL Server."""
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=yes;",
            timeout=5
        )

        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{TABLE_NAME}'
        """)
        columns = [row.COLUMN_NAME for row in cursor.fetchall()]
        conn.close()

        print("🔍 Loaded SQL Columns:", columns)
        return columns

    except Exception as e:
        print(f"⚠ SQL load failed: {e}")
        return []


# Lazy-loaded global
SQL_COLUMNS = []


def get_sql_columns():
    global SQL_COLUMNS
    if not SQL_COLUMNS:
        SQL_COLUMNS = load_sql_columns()
    return SQL_COLUMNS


# --------------------------
# GROQ CLIENT
# --------------------------
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("Missing GROQ_API_KEY")

    return Groq(api_key=api_key)


# --------------------------
# NLP HELPERS
# --------------------------
METRIC_SYNONYMS = {
    "revenue": "REVAmount",
    "sales": "REVAmount",
    "total revenue": "REVAmount",
    "rev": "REVAmount",

    "cost": "CSTAmount",
    "expense": "CSTAmount",

    "profit": "JobProfit",
    "jobprofit": "JobProfit",
    "margin": "JobProfit"
}

TRANSPORT_KEYWORDS = ["SEA", "AIR", "ROAD", "RAIL", "COU", "FSA", "NOJ", "UNKNOWN"]


# --------------------------
# LLM PARSER
# --------------------------
def extract_query(question):
    client = get_client()  # lazy load
    SQL_COLUMNS = get_sql_columns()  # lazy load

    prompt = f"""
You are a JSON-only assistant.  
Return ONLY valid JSON — no text outside JSON.

Use only these SQL columns:
{SQL_COLUMNS}

Convert the user question into this JSON:
{{
  "metric": "<one SQL column or null>",
  "aggregation": "sum | avg | count | max | min",
  "year": 2024,
  "mode": "SEA"
}}

Rules:
- If no metric found → "metric": null
- If no year found → "year": null
- If no mode found → "mode": null

User Question: "{question}"
JSON:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=120
    )

    raw = response.choices[0].message.content.strip()
    print("\nRAW LLM JSON:\n", raw, "\n")

    # Safe parse
    try:
        parsed = json.loads(raw)
    except:
        parsed = {"metric": None, "aggregation": "sum", "year": None, "mode": None}

    # --------------------------
    # FIX METRIC IF NULL
    # --------------------------
    if not parsed.get("metric") or parsed["metric"] not in SQL_COLUMNS:
        q = question.lower()
        for k, col in METRIC_SYNONYMS.items():
            if k in q:
                parsed["metric"] = col
                break

    # --------------------------
    # FIX MODE
    # --------------------------
    if not parsed.get("mode"):
        for m in TRANSPORT_KEYWORDS:
            if m.lower() in question.lower():
                parsed["mode"] = m
                break

    # Final validation
    if parsed.get("metric") not in SQL_COLUMNS:
        parsed["metric"] = None

    return parsed
