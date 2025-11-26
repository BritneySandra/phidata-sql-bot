# sql_runner.py - execute SQL and return rows (SAFE, minimal-change)
import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Helper: create a DB connection (same connection string shape as you already use)
def _get_connection():
    return pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};"
        f"PWD={os.getenv('SQL_PASSWORD')};"
        "Encrypt=no;TrustServerCertificate=yes;",
        timeout=10,
    )

def _normalize_params(params):
    """
    Return params unchanged except:
      - Convert empty strings to None
      - Keep numeric types as-is (do NOT uppercase strings)
    This avoids changing types that can affect SQL matching.
    """
    if not params:
        return None
    out = []
    for p in params:
        if p is None:
            out.append(None)
        elif isinstance(p, str) and p.strip() == "":
            out.append(None)
        else:
            out.append(p)
    return out

def run_sql(sql, params=None):
    """
    Execute parameterized SQL and return rows as list-of-dicts.

    - Uses pandas.read_sql which leverages pyodbc parameter binding reliably.
    - Returns [] if no rows.
    - Converts NaN -> None so JSON encoding works.
    """
    conn = None
    try:
        conn = _get_connection()
    except Exception as e:
        # Connection failure -> propagate a friendly error
        raise RuntimeError(f"DB connection error: {e}")

    cleaned = _normalize_params(params)

    try:
        # pandas will handle parameter substitution safely for pyodbc
        df = pd.read_sql(sql, conn, params=cleaned)
    except Exception as e:
        # Close connection and raise so upstream can capture the error
        try:
            conn.close()
        except Exception:
            pass
        raise RuntimeError(f"SQL execution error: {e}")

    try:
        conn.close()
    except Exception:
        pass

    if df is None or df.empty:
        return []

    # Replace NaN with None to be JSON serializable, convert numpy types to native Python
    df = df.where(pd.notnull(df), None)

    # convert Decimal/numpy types to Python native using to_dict orient records
    records = df.to_dict(orient="records")

    # Ensure numeric types are plain Python types (pandas usually handles this)
    normalized = []
    for r in records:
        newr = {}
        for k, v in r.items():
            # convert numpy int/float to native
            if hasattr(v, "item"):
                try:
                    newr[k] = v.item()
                    continue
                except Exception:
                    pass
            newr[k] = v
        normalized.append(newr)

    return normalized
