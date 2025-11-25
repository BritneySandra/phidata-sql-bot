# main.py — FINAL WORKING VERSION (updated for new sql_runner + debugging)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import extract_query, get_schema
from sql_builder import build_sql_from_plan
from sql_runner import run_sql

app = FastAPI()

# Allow Power BI iframe / external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TABLE = "WBI_BI_Data_V2"


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ---------------------------------------------------------
# Web UI (dark ChatGPT-style)
# ---------------------------------------------------------
@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return """
<!DOCTYPE html>
<html>
<head>
<title>PhiData SQL Chatbot</title>
<style>
body {
    background: #0d1117;
    color: #e6edf3;
    font-family: Arial;
    padding: 25px;
}
#chat-container { max-width: 900px; margin: auto; }
h2 { color: #58a6ff; }
#q {
    width: 75%; padding: 12px; border-radius: 6px;
    border: 1px solid #30363d; background: #161b22; color: white;
}
button {
    padding: 12px 20px; background: #238636; border: none;
    border-radius: 6px; color: white; cursor: pointer;
}
button:hover { background: #2ea043; }
.chat-box {
    margin-top: 20px; padding: 20px;
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    white-space: pre-wrap; overflow-x: auto;
}
.result-table {
    width: 100%; border-collapse: collapse; margin-top: 12px;
}
.result-table th, .result-table td {
    border: 1px solid #30363d; padding: 8px;
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
    let question = document.getElementById("q").value;
    if (!question) return;

    document.getElementById("a").innerHTML = "<i>Thinking...</i>";

    let res = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({question})
    });

    let data = await res.json();

    let html = "<b>SQL:</b><br>" + (data.sql || "(none)") + "<br><br>";

    if (data.params) {
        html += "<b>Params:</b><br>" + JSON.stringify(data.params) + "<br><br>";
    }

    if (data.rows && data.rows.length > 0 && data.columns.length > 0) {
        html += "<b>Result:</b><br>";
        html += "<table class='result-table'><tr>";
        data.columns.forEach(c => html += "<th>" + c + "</th>");
        html += "</tr>";

        data.rows.forEach(r => {
            html += "<tr>";
            data.columns.forEach(c => html += "<td>" + (r[c] ?? '') + "</td>");
            html += "</tr>";
        });

        html += "</table>";
    } else {
        html += "<b>Answer:</b><br>" + (data.result || "No data");
    }

    document.getElementById("a").innerHTML = html;
}
</script>

</body>
</html>
    """


# ---------------------------------------------------------
# ASK API Endpoint
# ---------------------------------------------------------
class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask_api(q: Query):
    schema = get_schema()

    try:
        # 1) Convert natural-language question → LLM JSON plan
        plan = extract_query(q.question)

        if "error" in plan:
            return {
                "sql": None,
                "params": None,
                "result": f"Plan error: {plan['error']}",
                "rows": [],
                "columns": []
            }

        # 2) Build SQL from plan
        sql, params, columns = build_sql_from_plan(plan, TABLE, schema)

        # 3) Execute SQL
        rows = run_sql(sql, params)

        # 4) Auto summary for single-value queries
        if len(rows) == 1 and len(columns) == 1:
            val = rows[0][columns[0]]
            try:
                val_fmt = f"{val:,}"
            except:
                val_fmt = str(val)
            summary = f"{columns[0]} = {val_fmt}"
        else:
            summary = f"{len(rows)} rows returned."

        return {
            "sql": sql,
            "params": params,
            "result": summary,
            "rows": rows,
            "columns": columns
        }

    except Exception as e:
        # Full debug output, safe for UI
        return JSONResponse(
            status_code=200,
            content={
                "sql": locals().get("sql", None),
                "params": locals().get("params", None),
                "result": f"SQL execution error: {e}",
                "rows": [],
                "columns": []
            }
        )
