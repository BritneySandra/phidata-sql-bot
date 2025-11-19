# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import extract_query, get_schema
from sql_runner import run_sql

app = FastAPI()

# CORS for Power BI iframe
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
                // Present result neatly
                if (data.rows) {
                    // build table
                    let html = "SQL:\\n" + data.sql + "\\n\\nResult:\\n";
                    html += data.rows.map(r => Object.values(r).join(" | ")).join("\\n");
                    document.getElementById("a").innerText = html;
                } else {
                    document.getElementById("a").innerText = "SQL:\\n" + data.sql + "\\n\\nAnswer:\\n" + data.result;
                }
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
    time = parsed.get("time", {})
    group_by = parsed.get("group_by", False)
    group_col = parsed.get("group_column")
    compare = parsed.get("compare", [])

    # Build WHERE and params from time filters
    where_clauses = []
    params = []

    if time.get("year"):
        where_clauses.append("FinancialYear = ?")
        params.append(time["year"])
    if time.get("quarter"):
        where_clauses.append("FinancialQuarter = ?")
        params.append(time["quarter"])
    if time.get("month"):
        where_clauses.append("FinancialMonth = ?")
        params.append(time["month"])

    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

    agg_map = {"sum":"SUM","avg":"AVG","max":"MAX","min":"MIN","count":"COUNT"}
    agg_sql = agg_map.get(agg,"SUM")

    # If group_by requested & group_col available -> run GROUP BY query
    schema = get_schema()
    if group_by and group_col:
        # ensure group_col exists in schema
        if group_col not in schema:
            return JSONResponse({"sql": None, "result": f"❌ Unknown group column {group_col}"})
        sql = f"""
            SELECT {group_col} AS category, {agg_sql}([{metric}]) AS value
            FROM {TABLE}
            WHERE {where_clause}
            GROUP BY {group_col}
            ORDER BY {group_col}
        """
        rows = run_sql(sql, params)
        # format result rows
        if not rows:
            return {"sql": sql, "result": "No data found", "rows": []}
        # return structured rows and a printable table
        return {"sql": sql, "result": f"{len(rows)} rows", "rows": rows}

    # If user asked to compare multiple category values (e.g., compare sea and air)
    if compare and group_col:
        placeholders = ",".join("?" for _ in compare)
        where_with_compare = where_clause + f" AND {group_col} IN ({placeholders})" if where_clause != "1=1" else f"{group_col} IN ({placeholders})"
        sql = f"""
            SELECT {group_col} AS category, {agg_sql}([{metric}]) AS value
            FROM {TABLE}
            WHERE {where_with_compare}
            GROUP BY {group_col}
            ORDER BY {group_col}
        """
        params_full = params + compare
        rows = run_sql(sql, params_full)
        if not rows:
            return {"sql": sql, "result": "No data found", "rows": []}
        return {"sql": sql, "result": f"{len(rows)} rows", "rows": rows}

    # Otherwise single aggregate
    sql = f"""
        SELECT {agg_sql}([{metric}]) AS value
        FROM {TABLE}
        WHERE {where_clause}
    """
    rows = run_sql(sql, params)
    if not rows or rows[0].get("value") is None:
        return {"sql": sql, "result": "No data found"}
    value = rows[0]["value"]
    result_text = f"{agg} of {metric} is {value:,}"
    if time.get("year"):
        result_text += f" in {time['year']}"
    if group_col:
        result_text += f" grouped by {group_col}"
    return {"sql": sql, "result": result_text}
