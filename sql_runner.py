# sql_runner.py - execute SQL and return rows
import pyodbc
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def run_sql(sql, params=None):
    """
    Execute SQL safely and return rows as list of dicts.
    Uses pandas.read_sql with params for parameterized query.
    """
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};"
        f"PWD={os.getenv('SQL_PASSWORD')};"
        f"Encrypt=no;TrustServerCertificate=yes;"
    )

    # pandas expects params sequence/tuple for pyodbc
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()

    # If DataFrame empty, return []
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")
