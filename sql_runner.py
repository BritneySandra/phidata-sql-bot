# sql_runner.py
import pyodbc
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def run_sql(sql, params=None):
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};"
        f"PWD={os.getenv('SQL_PASSWORD')};"
        "Encrypt=no;TrustServerCertificate=yes;"
    )

    cleaned = []
    if params:
        for p in params:
            if isinstance(p, str):
                cleaned.append(p.strip())
            else:
                cleaned.append(p)

    try:
        df = pd.read_sql(sql, conn, params=cleaned)
    finally:
        conn.close()

    if df.empty:
        return []
    return df.to_dict(orient="records")
