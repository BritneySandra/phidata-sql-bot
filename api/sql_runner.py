import pyodbc
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def run_sql(sql, params=None):
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};"
        f"PWD={os.getenv('SQL_PASSWORD')};"
    )
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df.to_dict(orient="records")
