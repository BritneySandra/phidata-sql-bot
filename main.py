# ---------------------------------------------------------
# main.py  (FINAL VERSION)
# ---------------------------------------------------------

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import extract_query, get_sql_columns
from sql_runner import run_sql

app = FastAPI()

# ----- CORS (Required for Power BI) -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TABLE = "WBI_BI_Data_V2"


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ---------------------------------------------------------
# SIMPLE WEB CHAT UI
# ---------------------------------------------------------
@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return """
    <html>
    <body style="font-family: Arial; padding: 20px;">
        <h2>PhiData SQL Chatbot</h2>
        <input id="q" style="width:80%;padding:8px;" placeholder="Ask any question">
        <button onclick="ask()">Send</button>
        <pre id="a" style="margin-top:20px;background:#eee;padding:10px;"></pre>

        <script>
            async function ask() {
                let q = document.getElementById("q").value;
                let res = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({question: q})
                });
                let data = await res.json();
                document.getElementById("a").innerText =
                    "SQL:\\n" + data.sql +
                    "\\n\\nAnswer:\\n" + data.result;
            }
        </script>
    </body>
    </html>
    """


# ---------------------------------------------------------
# ASK ENDPOINT
# ---------------------------------------------------------
class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(q: Query):

    # Load columns
    SQL_COLUMNS = get_sql_columns()

    parsed = extract_query(q.question)

    metric = parsed.get("metric")
    agg = parsed.get("aggregation", "sum")
    year = parsed.get("year")
    mode = parsed.get("mode")

    # Build SQL
    agg_sql = {
        "sum": "SUM",
        "avg": "AVG",
        "max": "MAX",
        "min": "MIN",
        "count": "COUNT"
    }.get(agg, "SUM")

    where = []
    params = []

    if mode:
        where.append("TransportMode = ?")
        params.append(mode)

    if year:
        where.append("FinancialYear = ?")
        params.append(year)

    where_clause = " AND ".join(where) if where else "1=1"

    sql = f"""
        SELECT {agg_sql}([{metric}]) AS value
        FROM {TABLE}
        WHERE {where_clause}
    """

    rows = run_sql(sql, params)

    if not rows or rows[0]["value"] is None:
        return {"sql": sql, "result": "No data found"}

    value = rows[0]["value"]

    # Human-readable answer
    answer = f"{agg} of {metric} is {value:,}"
    if mode:
        answer += f" for {mode}"
    if year:
        answer += f" in {year}"

    return {"sql": sql, "result": answer}
