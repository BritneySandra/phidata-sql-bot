# ===========================
# main.py (FINAL UPDATED)
# ===========================
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
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# UI
@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return """
<html><body style="background:#0d1117;color:white;font-family:Arial;padding:20px;">
<h2>PhiData SQL Chatbot</h2>
<input id='q' style='width:80%;padding:10px;border-radius:5px;background:#161b22;color:white;'>
<button onclick='ask()' style='padding:10px 20px;background:#238636;color:white;border:none;'>Send</button>
<div id='a' style='white-space:pre-wrap;margin-top:20px;background:#161b22;padding:15px;'></div>

<script>
async function ask(){
    let q=document.getElementById("q").value;
    document.getElementById("a").innerHTML="Thinking...";

    let res=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({question:q})});

    let data=await res.json();

    let html="<b>SQL:</b><br>"+(data.sql||"(none)")+"<br><br>";

    if(data.rows && data.rows.length){
        html+="<b>Result:</b><br><table border='1' cellpadding='5'>";
        html+="<tr>"+data.columns.map(c=>"<th>"+c+"</th>").join('')+"</tr>";
        data.rows.forEach(r=>{
            html+="<tr>"+data.columns.map(c=>"<td>"+(r[c]||'')+"</td>").join('')+"</tr>";
        });
        html+="</table>";
    } else {
        html+="<b>Answer:</b><br>"+data.result;
    }

    document.getElementById("a").innerHTML=html;
}
</script>
</body></html>
"""

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    try:
        plan = extract_query(q.question)

        sql, params, columns = build_sql_from_plan(plan, TABLE, get_schema())

        rows = run_sql(sql, params)

        if len(rows)==1 and len(columns)==1:
            val = rows[0][columns[0]]
            try: val_fmt = f"{val:,}"
            except: val_fmt = val
            summary = f"{columns[0]} = {val_fmt}"
        else:
            summary = f"{len(rows)} rows returned."

        return {"sql": sql, "rows": rows, "columns": columns, "result": summary}

    except Exception as e:
        return JSONResponse(content={
            "sql": None,
            "rows": [],
            "columns": [],
            "result": f"Error: {e}"
        })
