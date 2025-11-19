# ---------------------------------------------------------
# agent.py  (FINAL VERSION - handles ANY user question)
# ---------------------------------------------------------

from groq import Groq
import os
import json
import pyodbc
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "WBI_BI_Data_V2"


# ---------------------------------------------------------
# LOAD SQL COLUMNS (Lazy)
# ---------------------------------------------------------
def load_sql_columns():
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            f"Encrypt=no;"
            f"TrustServerCertificate=yes;",
        )
        cursor = conn.cursor()
        cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{TABLE_NAME}'")
        cols = [row.COLUMN_NAME for row in cursor.fetchall()]
        conn.close()
        return cols
    except Exception as e:
        print("❌ Column load failed:", e)
        return []


SQL_COLUMNS = []


def get_sql_columns():
    global SQL_COLUMNS
    if not SQL_COLUMNS:
        SQL_COLUMNS = load_sql_columns()
    return SQL_COLUMNS


# ---------------------------------------------------------
# LLM CLIENT
# ---------------------------------------------------------
def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise Exception("Missing GROQ_API_KEY")
    return Groq(api_key=key)


# ---------------------------------------------------------
# SYNONYMS & NLP HELPERS
# ---------------------------------------------------------
METRIC_SYNONYMS = {
    "revenue": "REVAmount",
    "total revenue": "REVAmount",
    "sales": "REVAmount",
    "income": "REVAmount",
    "turnover": "REVAmount",

    "cost": "CSTAmount",
    "expense": "CSTAmount",
    "spend": "CSTAmount",

    "profit": "JobProfit",
    "margin": "JobProfit",
    "earnings": "JobProfit"
}

TRANSPORT_KEYWORDS = ["SEA", "AIR", "ROAD", "RAIL", "COU", "FSA", "NOJ", "UNKNOWN"]


# ---------------------------------------------------------
# MAIN EXTRACTION LOGIC
# ---------------------------------------------------------
def extract_query(question):

    SQL_COLUMNS = get_sql_columns()
    client = get_client()

    # ---- LLM Prompt ----
    prompt = f"""
You are a JSON-only assistant (STRICT).
Return ONLY valid JSON. No explanation.

Using only these SQL columns:
{SQL_COLUMNS}

Convert the user question into:

{{
  "metric": "column-name or null",
  "aggregation": "sum | avg | max | min | count",
  "year": 2024,
  "mode": "SEA"
}}

RULES:
- If metric not found → metric = null
- If aggregation unclear → aggregation = "sum"
- If year not in question → year = null
- If mode not in question → mode = null

User question: "{question}"

Return ONLY JSON:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    # Safe parse
    try:
        parsed = json.loads(raw)
    except:
        parsed = {"metric": None, "aggregation": "sum", "year": None, "mode": None}

    # ---------------------------------------------------------
    # FALLBACK 1: Metric Correction
    # ---------------------------------------------------------
    q_lower = question.lower()

    if not parsed.get("metric") or parsed["metric"] not in SQL_COLUMNS:
        for syn, col in METRIC_SYNONYMS.items():
            if syn in q_lower:
                parsed["metric"] = col
                break

    # Still null? Choose BEST COLUMN automatically
    if parsed.get("metric") not in SQL_COLUMNS:
        parsed["metric"] = "REVAmount"     # default best guess

    # ---------------------------------------------------------
    # FALLBACK 2: Mode Detection
    # ---------------------------------------------------------
    if not parsed.get("mode"):
        for m in TRANSPORT_KEYWORDS:
            if m.lower() in q_lower:
                parsed["mode"] = m
                break

    return parsed
