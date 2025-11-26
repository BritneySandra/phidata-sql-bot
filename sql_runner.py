# sql_runner.py - execute SQL and return rows
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

    # 🔥 FIX: CLEAN PARAMS BEFORE EXECUTION
    clean_params = []
    if params:
        for p in params:
            if p is None:
                clean_params.append(None)
            elif isinstance(p, str):
                clean_params.append(p.strip().upper())   # normalize to uppercase
            else:
                clean_params.append(p)

    try:
        df = pd.read_sql(sql, conn, params=clean_params)
    finally:
        conn.close()

    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")

