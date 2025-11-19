from groq import Groq
import os
import json
import pyodbc
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"


def load_sql_columns():
    """Load column names dynamically from the SQL table."""
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};",
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
        return columns

    except Exception as e:
        print(f"⚠ SQL load failed: {e}")
        return []


# ❗ DO NOT LOAD DURING IMPORT
SQL_COLUMNS = []


def get_sql_columns():
    """Lazy-load columns when needed."""
    global SQL_COLUMNS
    if not SQL_COLUMNS:
        SQL_COLUMNS = load_sql_columns()
    return SQL_COLUMNS


def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("Missing GROQ_API_KEY")
    return Groq(api_key=api_key)


METRIC_SYNONYMS = {
    "revenue": "REVAmount",
    "total revenue": "REVAmount",
    "sales": "REVAmount",

    "cost": "CSTAmount",
    "expense": "CSTAmount",

    "profit": "JobProfit",
    "margin": "JobProfit"
}

TRANSPORT_KEYWORDS = ["SEA", "AIR", "ROAD", "RAIL", "COU", "FSA", "NOJ", "Unknown"]


def extract_query(question):
    client = get_client()
    SQL_COLUMNS = get_sql_columns()

    prompt = f"""
You are a strict JSON generator.

ONLY output valid JSON.

Use these SQL columns: {SQL_COLUMNS}

Convert the question into JSON:
{{
  "metric": "<one column>",
  "aggregation": "sum | avg | max | min | count",
  "year": 2024,
  "mode": "SEA"
}}

Rules:
- If metric not found → "metric": null
- If year not found → "year": null
- If mode not found → "mode": null

User Question: "{question}"
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )

    text = response.choices[0].message.content.strip()

    # Safe parse
    try:
        parsed = json.loads(text)
    except:
        parsed = {"metric": None, "aggregation": "sum", "year": None, "mode": None}

    # Fix metric using synonyms
    if not parsed.get("metric") or parsed["metric"] not in SQL_COLUMNS:
        q = question.lower()
        for k, col in METRIC_SYNONYMS.items():
            if k in q:
                parsed["metric"] = col
                break

    # Fix mode
    if not parsed.get("mode"):
        for m in TRANSPORT_KEYWORDS:
            if m.lower() in question.lower():
                parsed["mode"] = m
                break

    # Final validation
    if parsed["metric"] not in SQL_COLUMNS:
        parsed["metric"] = None

    return parsed
