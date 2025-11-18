from groq import Groq
import os
import json
import pyodbc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"

def load_sql_columns():
    """Load column names dynamically from the SQL table."""
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
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
        print(f"❌ Failed to load columns: {e}")
        return []

# Load columns dynamically
SQL_COLUMNS = load_sql_columns()
print("📊 Loaded SQL Columns:", SQL_COLUMNS)

# Synonyms for metrics
METRIC_SYNONYMS = {
    "revenue": "REVAmount",
    "total revenue": "REVAmount",
    "sales": "REVAmount",
    "rev": "REVAmount",

    "cost": "CSTAmount",
    "total cost": "CSTAmount",
    "expense": "CSTAmount",

    "profit": "JobProfit",
    "jobprofit": "JobProfit",
    "margin": "JobProfit"
}

# Transport mode keywords
TRANSPORT_KEYWORDS = ["SEA", "AIR", "ROAD", "RAIL", "COU", "FSA", "NOJ", "Unknown"]


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_query(question):
    prompt = f"""
You are a strict JSON generator.

ONLY OUTPUT VALID JSON.
DO NOT OUTPUT SQL.
DO NOT OUTPUT CODE BLOCKS.
DO NOT OUTPUT ANY TEXT BEFORE OR AFTER JSON.

Use only these SQL columns: {SQL_COLUMNS}

Convert the user question into this JSON structure:
{{
  "metric": "<one SQL column>",
  "aggregation": "sum | avg | count | max | min",
  "year": 2024,
  "mode": "SEA"
}}

Rules:
- If metric not found → set "metric": null
- If year not found → set "year": null
- If transport mode not found → set "mode": null

User Question: "{question}"

Output JSON only:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )

    text = response.choices[0].message.content.strip()
    print("\nRAW LLM JSON:\n", text, "\n")

    # safe parse
    try:
        parsed = json.loads(text)
    except:
        parsed = {"metric": None, "aggregation": "sum", "year": None, "mode": None}

    # Semantic Metric Fix — If model returns null or natural text
    if not parsed.get("metric") or parsed["metric"] not in SQL_COLUMNS:
        q_lower = question.lower()
        for keyword, col in METRIC_SYNONYMS.items():
            if keyword in q_lower:
                parsed["metric"] = col
                break

    # Fix mode detection (SEA, AIR, etc.)
    if not parsed.get("mode"):
        for m in TRANSPORT_KEYWORDS:
            if m.lower() in question.lower():
                parsed["mode"] = m
                break

    # Final validation
    if parsed.get("metric") not in SQL_COLUMNS:
        parsed["metric"] = None

    return parsed
