# sql_runner.py - safe SQL executor
import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()

def run_sql(sql: str, params=None):
    """
    Execute SQL and return:
    {
        "columns": [...],
        "rows": [ [...], [...], ... ]
    }
    Always JSON-serializable.
    """
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

        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)

        # Get column names
        columns = [c[0] for c in cur.description]

        # Convert all row objects → Python lists
        raw_rows = cur.fetchall()
        rows = [list(r) for r in raw_rows]

        conn.close()
        return {
            "columns": columns,
            "rows": rows
        }

    except Exception as e:
        return {
            "error": str(e),
            "sql": sql
        }
