# sql_runner.py - execute SQL and return rows
import pyodbc
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def run_sql(sql, params=None):
    """
    Executes a SQL query and returns (columns, rows)
    Returns (None, error_message) when SQL fails.
    """
    conn = None
    try:
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

        # Execute with parameters
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)

        rows = cur.fetchall()

        # Extract column names
        cols = [column[0] for column in cur.description]

        return cols, rows

    except Exception as e:
        return None, str(e)

    finally:
        if conn:
            conn.close()
