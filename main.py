from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ❗ Do not import SQL columns or Groq client at import time
from agent import extract_query, get_sql_columns
from sql_runner import run_sql

app = FastAPI()

TABLE = "WBI_BI_Data_V2"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/docs-check")
async def docs_check():
    return "FastAPI is running."


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return """... (same HTML) ..."""


class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask(q: Query):

    SQL_COLUMNS = get_sql_columns()  # lazy load

    parsed = extract_query(q.question)
    metric = parsed.get("metric")

    if metric not in SQL_COLUMNS:
        return {"sql": None, "result": "❌ Unknown metric"}

    # (rest of the logic unchanged)
