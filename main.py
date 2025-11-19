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


# ---------------------------------------------------------------------
# 🔥 DARK MODE CHATGPT STYLE CHAT UI  (Power BI compatible)
# ---------------------------------------------------------------------
@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return """
<!DOCTYPE html>
<html>
<head>
<title>PhiData SQL Chatbot</title>

<style>
body {
    background-color: #0d1117;
    font-family: Arial, sans-serif;
    color: #e6edf3;
    margin: 0;
    padding: 20px;
}
#chat-container {
    max-width: 900px;
    margin: auto;
}
h2 {
    color: #58a6ff;
    margin-bottom: 20px;
}
#q {
    width: 80%;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid #30363d;
    background: #161b22;
    color: white;
    outline: none;
}
#q::placeholder {
    color: #8b949e;
}
button {
    padding: 12px 20px;
    background: #238636;
    border: none;
    color: white;
    border-radius: 6px;
    cursor: pointer;
}
button:hover {
    background: #2ea043;
}
.chat-box {
    background: #161b22;
    border: 1px solid #30363d;
    padding: 15px;
    margin-top: 20px;
    border-radius: 8px;
    white-space: pre-wrap;
    font-size: 14px;
    overflow-x: auto;
}
.result-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}
.result-table th, .result-table td {
    border: 1px solid #30363d;
    padding: 8px;
    background: #0d1117;
}
</style>

</head>
<body>
<div id="chat-container">
    <h2>PhiData SQL Chatbot</h2>

    <input id="q" placeholder="Ask any question...">
    <button onclick="ask()">Send</button>

    <div id="a" class="chat-box"></div>
</div>


<script>
async function ask() {
    let q = document.getElementById("q").value;
    document.getElementById("a").innerHTML = "<i>Thinking...</i>";

    let res = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({question: q})
    });

    let data = await res.json();

    let html = "<b>SQL:</b><br>" + data.sql + "<br><br>";

    // If table rows exist → show proper table
    if (data.rows) {
        html += "<b>Result:</b><br>";
        html += "<table class='result-table'>";

        if (data.columns) {
            html += "<tr>";
            data.columns.forEach(c => html += "<th>" + c + "</th>");
            html += "</tr>";
        }

        data.rows.forEach(r => {
            html += "<tr>";
            Object.values(r).forEach(v => html += "<td>" + v + "</td>");
            html += "</tr>";
        });

        html += "</table>";
    } else {
        html += "<b>Answer:</b><br>" + data.result;
    }

    document.getElementById("a").innerHTML = html;
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
    category_value = parsed.get("category_value")
    compare = parsed.get("compare", [])

    schema = get_schema()
    if not metric or metric not in schema:
        return JSONResponse({
            "sql": None,
            "result": "❌ Unknown metric (I couldn't find a numeric column)."
        })

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

    if group_col and category_value:
        where_clauses.append(f"{group_col} = ?")
        params.append(category_value)

    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

    agg_map = {"sum":"SUM","avg":"AVG","max":"MAX","min":"MIN","count":"COUNT"}
    agg_sql = agg_map.get(agg,"SUM")

    # CASE A → full category breakdown
    if group_by and group_col and not category_value and not compare:
        sql = f"""
            SELECT {group_col} AS category, {agg_sql}([{metric}]) AS value
            FROM {TABLE}
            WHERE {where_clause}
            GROUP BY {group_col}
            ORDER BY value DESC
        """
        rows = run_sql(sql, params)
        return {"sql": sql, "columns":["category","value"], "rows": rows}

    # CASE B → compare categories
    if compare and group_col:
        placeholders = ",".join("?" for _ in compare)
        where_plus = where_clause + " AND " if where_clause != "1=1" else ""
        where_plus += f"{group_col} IN ({placeholders})"

        sql = f"""
            SELECT {group_col} AS category, {agg_sql}([{metric}]) AS value
            FROM {TABLE}
            WHERE {where_plus}
            GROUP BY {group_col}
            ORDER BY value DESC
        """
        rows = run_sql(sql, params + compare)
        return {"sql": sql, "columns":[group_col,"value"], "rows": rows}

    # CASE C → specific category value
    if group_col and category_value:
        sql = f"""
            SELECT {agg_sql}([{metric}]) AS value
            FROM {TABLE}
            WHERE {where_clause}
        """
        rows = run_sql(sql, params)
        if not rows or rows[0].get("value") is None:
            return {"sql": sql, "result": "No data found"}

        value = rows[0]["value"]
        return {"sql": sql, "result": f"{metric} for {category_value} = {value:,}"}

    # CASE D → simple aggregate
    sql = f"""
        SELECT {agg_sql}([{metric}]) AS value
        FROM {TABLE}
        WHERE {where_clause}
    """
    rows = run_sql(sql, params)
    if not rows or rows[0].get("value") is None:
        return {"sql": sql, "result": "No data found"}

    value = rows[0]["value"]
    return {"sql": sql, "result": f"{agg} of {metric} is {value:,}"}
