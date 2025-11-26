# sql_runner.py - execute SQL and return rows
import pyodbc
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def run_sql_query(sql, params=None):
    import pyodbc

    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};"
        f"PWD={os.getenv('SQL_PASSWORD')};"
        f"Encrypt=no;TrustServerCertificate=yes;",
        timeout=10,
    )
    cur = conn.cursor()

    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)

        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]
        conn.close()

        return cols, rows

    except Exception as e:
        conn.close()
        return None, str(e)

