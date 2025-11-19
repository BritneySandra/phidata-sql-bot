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
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USERNAME')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            ,
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


# ❗ DO NOT LOAD COLUMNS DURING IMPORT
SQL_COLUMNS = []

def get_sql_columns():
    """Lazy-load columns when needed."""
    global SQL_COLUMNS
    if not SQL_COLUMNS:
        SQL_COLUMNS = load_sql_columns()
    return SQL_COLUMNS


# Safe Groq initialization
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("Missing GROQ_API_KEY")
    return Groq(api_key=api_key)


METRIC_SYNONYMS = {
    "revenue": "REVAmount",
    "cost": "CSTAmount",
    "profit": "JobProfit"
}

TRANSPORT_KEYWORDS = ["SEA", "AIR", "ROAD", "RAIL", "COU", "FSA", "NOJ", "Unknown"]


def extract_query(question):
    client = get_client()  # lazy load
    SQL_COLUMNS = get_sql_columns()  # lazy load
