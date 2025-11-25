import pyodbc
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def run_sql(sql, params=None):
    """Execute SQL safely and return rows as list of dicts."""

    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};"
        f"PWD={os.getenv('SQL_PASSWORD')};"
        f"Encrypt=no;"
        f"TrustServerCertificate=yes;"
    )

    df = pd.read_sql(sql, conn, params=params)
    conn.close()

    return df.to_dict(orient="records")
