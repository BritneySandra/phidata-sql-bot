from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy imports
from agent import extract_query, get_sql_columns
from sql_runner import run_sql

# ----------------------------------------------------
# APP + CORS  (REQUIRED for Power BI POST + OPTIONS)
# ----------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Power BI Desktop allows iframe only with *
    allow_credentials=True,
    allow_methods=["*"],            # Enables POST + OPTIONS
    allow_headers=["*"],            # Allows JSON headers
)

TABLE = "WBI_BI_Data_V2"

# ----------------------------------------------------
# HEALTH CHECKS
# ----------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/docs-check")
async def docs_check():
    return {"status": "running", "message": "FastAPI is ready"}

# ----------------------------------------------------
# SIMPLE CHAT UI (Power BI iframe loads this)
# ----------------------------------------------------

@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return """
    <html>
    <body style="font-family: Arial; padding: 20px;">
        <h2>PhiData SQL Chatbot</h2>

        <input id="question" style="width:80%;padding:8px;" placeholder="Ask something...">
        <button onclick="ask()">Send</button>

        <pre id="answer" style="margin-top:20px;background:#eee;padding:10px;"></pre>

        <script>
            async function ask() {
                let q = document.getElementById("question").value;

                let res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q })
                });

                let data = await res.json();

                document.getElementById("answer").innerText =
                    "SQL Generated:\\n" + data.sql +
                    "\\n\\nResult:\\n" + data.result;
            }
        </script>
    </body>
    </html>
    """

# ----------------------------------------------------
# ASK ENDPOINT
# ----------------------------------------------------

class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(q: Query):

    # Load columns dynamically
    SQL_COLUMNS = get_sql_columns()

    parsed = extract_query(q.question)
    metric = parsed.get("metric")
    agg = parsed.get("aggregation", "sum")
    year = parsed.get("year")
    modes = parsed.get("modes") or []

    if metric not in SQL_COLUMNS:
        return {"sql": None, "result": "❌ Unknown metric"}

    agg_sql = {
        "sum": "SUM",
        "avg": "AVG",
        "max": "MAX",
        "min": "MIN",
        "count": "COUNT"
    }.get(agg, "SUM")

    where = []
    params = {}

    # Transport Modes
    if len(modes) > 0:
        where.append(f"TransportMode IN ({','.join(['?' for _ in modes])})")
        for idx, m in enumerate(modes):
            params[idx] = m

    # Year
    if year is not None:
        where.append("FinancialYear = ?")
        params[len(params)] = year

    where_clause = " AND ".join(where) if where else "1=1"

    sql = f"""
        SELECT {agg_sql}([{metric}]) AS value
        FROM {TABLE}
        WHERE {where_clause}
    """

    rows = run_sql(sql, list(params.values()))

    if not rows or "value" not in rows[0]:
        return {"sql": sql, "result": "No data found"}

    value = rows[0]["value"]

    # Generate friendly text response
    if len(modes) == 1:
        result_text = f"{metric} for {modes[0]} is {value:,}"
    elif len(modes) > 1:
        result_text = f"Comparison result: {value:,}"
    elif year:
        result_text = f"{metric} for {year} is {value:,}"
    else:
        result_text = f"Total {metric} across all years is {value:,}"

    return {
        "sql": sql,
        "result": result_text
    }
