# main.py - FastAPI endpoints and UI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import extract_query, get_schema
from sql_builder import build_sql_from_plan
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
<!DOCTYPE html>
<html>
<head><title>PhiData SQL Chatbot</title>
<style>
body { background-color: #0d1117; font-family: Arial, sans-serif; color: #e6edf3; margin: 0; padding: 20px; }
#chat-container { max-width: 900px; margin: auto; }
h2 { color: #58a6ff; margin-bottom: 20px; }
#q { width: 80%; padding: 12px; border-radius: 6px; border: 1px solid #30363d; background: #161b22; color: white; outline: none; }
#q::placeholder { color: #8b949e; }
button { padding: 12px 20px; background: #238636; border: none; color: white; border-radius: 6px; cursor: pointer; }
button:hover { background: #2ea043; }
.chat-box { background: #161b22; border: 1px solid #30363d; padding: 15px; margin-top: 20px; border-radius: 8px; white-space: pre-wrap; font-size: 14px; overflow-x: auto; }
.result-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
.result-table th, .result-table td { border: 1px solid #30363d; padding: 8px; background: #0d1117; }
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
    if (!q) return;
    document.getElementById("a").innerHTML = "<i>Thinking...</i>";
    let res = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({question: q})
    });
    let data = await res.json();
    let html = "<b>SQL:</b><br>" + (data.sql || "(none)") + "<br><br>";
    if (data.rows && data.rows.length > 0 && data.columns) {
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

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    schema = get_schema()
    try:
        plan = extract_query(q.question)

        # If AI returned error, forward it
        if isinstance(plan, dict) and plan.get("error"):
            return {"sql": None, "result": f"AI returned incomplete plan: {plan.get('error')}", "rows": [], "columns": []}

        sql, params, columns = build_sql_from_plan(plan, TABLE, schema)

        rows = run_sql(sql, params)

        if not rows:
            return {"sql": sql, "result": "No data found", "rows": [], "columns": columns}

        # If scalar single-row single-col, return human-friendly sentence
        if len(rows) == 1 and len(columns) == 1:
            val = rows[0].get(columns[0])
            try:
                # numeric formatting with commas (if numeric)
                if isinstance(val, (int, float)):
                    val_fmt = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
                else:
                    val_fmt = str(val)
            except Exception:
                val_fmt = str(val)
            # produce a natural language response
            summary = f"The {columns[0].replace('_',' ')} is {val_fmt}."
        else:
            summary = f"{len(rows)} rows returned."

        return {"sql": sql, "result": summary, "rows": rows, "columns": columns}

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "sql": locals().get("sql", None),
                "result": f"SQL execution error: {e}",
                "rows": [],
                "columns": []
            }
        )
