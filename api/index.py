from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from agent import extract_query, SQL_COLUMNS
from sql_runner import run_sql

app = FastAPI()

TABLE = "WBI_BI_Data_V2"

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
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({question: q})
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

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):

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

    # TRANSPORT MODE LOGIC
    if len(modes) > 0:
        where.append(f"TransportMode IN ({','.join(['?' for _ in modes])})")
        for idx, m in enumerate(modes):
            params[idx] = m

    # YEAR LOGIC
    if year is not None:
        where.append("FinancialYear = ?")
        params[len(params)] = year

    where_clause = " AND ".join(where) if where else "1=1"

    # AGG QUERY
    sql = f"""
        SELECT {agg_sql}([{metric}]) AS value
        FROM {TABLE}
        WHERE {where_clause}
    """

    rows = run_sql(sql, list(params.values()))

    # PROCESS RESULT
    if not rows or "value" not in rows[0]:
        return {"sql": sql, "result": "No data found"}

    result_value = rows[0]["value"]

    # RESPONSE RULE
    if len(modes) == 1:
        result = f"{metric} for {modes[0]} is {result_value:,}"

    elif len(modes) > 1:
        result = f"Comparison result: {result_value:,}"

    elif year:
        result = f"{metric} for {year} is {result_value:,}"

    else:
        result = f"Total {metric} across all years is {result_value:,}"

    return {
        "sql": sql,
        "result": result
    }
