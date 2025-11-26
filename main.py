# main.py - FastAPI endpoints and UI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import extract_query, get_schema
from sql_builder import build_sql_from_plan
from sql_runner import run_sql

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TABLE = "WBI_BI_Data_V2"

@app.get("/")
async def root():
    return {"status": "ok", "message": "API running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    schema = get_schema()
    try:
        plan = extract_query(q.question)

        if isinstance(plan, dict) and plan.get("error"):
            return {
                "sql": None,
                "result": f"Could not understand question: {plan.get('error')}",
                "rows": [],
                "columns": []
            }

        sql, params, columns = build_sql_from_plan(plan, TABLE, schema)
        rows = run_sql(sql, params)

        if not rows:
            return {"sql": sql, "result": "No data found", "rows": [], "columns": columns}

        if len(rows) == 1 and len(columns) == 1:
            val = rows[0].get(columns[0])
            if isinstance(val, (int, float)):
                val_fmt = f"{val:,.2f}"
            else:
                val_fmt = str(val)
            summary = f"The {columns[0].replace('_',' ')} is {val_fmt}."
        else:
            summary = f"Returned {len(rows)} row(s)."

        return {"sql": sql, "result": summary, "rows": rows, "columns": columns}

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "sql": locals().get("sql", None),
                "result": f"Execution error: {str(e)}",
                "rows": [],
                "columns": []
            }
        )
