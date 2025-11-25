# sql_runner.py — FINAL FIXED (parameterized, no pandas)

import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()

def run_sql(sql, params=None):
    """Execute SQL query safely with pyodbc parameter binding."""
    
    if params is None:
        params = []

    # Clean primitive parameters
    clean_params = []
    for p in params:
        if isinstance(p, dict):
            # Pick sensible value
            for k in ("value", "year", "month", "quarter"):
                if k in p:
                    p = p[k]
                    break
            else:
                p = str(p)
        clean_params.append(p)

    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};"
        f"PWD={os.getenv('SQL_PASSWORD')};"
        "Encrypt=no;TrustServerCertificate=yes;"
    )

    cursor = conn.cursor()

    # Use proper pyodbc parameter execution
    cursor.execute(sql, tuple(clean_params))

    columns = [col[0] for col in cursor.description] if cursor.description else []
    rows = []

    for row in cursor.fetchall():
        rows.append({columns[i]: row[i] for i in range(len(columns))})

    cursor.close()
    conn.close()

    return rows
