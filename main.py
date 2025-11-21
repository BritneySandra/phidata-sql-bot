# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import extract_query
from sql_builder import build_sql
from sql_runner import run_sql  # your existing DB executor

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


# ------------------------------------------------------
# Dark-mode ChatGPT-style UI (works in browser + Power BI)
# ------------------------------------------------------
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

    let html = "<b>SQL:</b><br>" + (data.sql || "(no SQL)") + "<br><br>";

    if (data.rows && data.rows.length > 0) {
        html += "<b>Result:</b><br>";
        html += "<table class='result-table'>";

        if (data.columns && data.columns.length) {
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
        html += "<b>Answer:</b><br>" + (data.result || "No data found");
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
    # 1) Use agent to convert question -> generic intent
    intent = extract_query(q.question)

    if not intent.get("select"):
        return JSONResponse(
            {"sql": None, "result": "I couldn't understand which measure to use."},
            status_code=200,
        )

    # 2) Build SQL from intent
    sql, params = build_sql(intent, TABLE)

    # 3) Execute SQL
    try:
        rows = run_sql(sql, params or [])
    except Exception as e:
        return JSONResponse(
            {"sql": sql, "result": f"SQL execution error: {e}", "rows": []},
            status_code=200,
        )

    if not rows:
        return {"sql": sql, "result": "No data found", "rows": []}

    # columns for UI table
    columns = list(rows[0].keys())

    # human-friendly summary for single scalar
    if len(rows) == 1 and len(columns) == 1:
        val = rows[0][columns[0]]
        result_text = f"{columns[0]} = {val:,}" if isinstance(val, (int, float)) else str(
            val
        )
    else:
        result_text = f"{len(rows)} rows"

    return {
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "result": result_text,
    }
