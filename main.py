# main.py - FastAPI endpoints and UI (with correct metric-based YOY logic)
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import extract_query, get_schema, METRICS, detect_metric_from_text
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

   if (data.rows && data.rows.length > 0) {
    html += "<b>Result:</b><br>";

    let cols = Object.keys(data.rows[0]);

    html += "<table class='result-table'><tr>";
    cols.forEach(c => html += "<th>" + c + "</th>");
    html += "</tr>";

    data.rows.forEach(r => {
        html += "<tr>";
        cols.forEach(c => html += "<td>" + (r[c] ?? '') + "</td>");
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


def _filters_to_where(filters, schema, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = set()

    where_clauses = []
    params = []

    for flt in filters:
        if not isinstance(flt, dict):
            continue
        col = flt.get("column")
        if not col or col in exclude_columns or col not in schema:
            continue
        op = (flt.get("operator") or "=").lower()
        val = flt.get("value")

        if op == "in" and isinstance(val, (list, tuple)):
            placeholders = ", ".join("?" for _ in val)
            where_clauses.append(f"[{col}] IN ({placeholders})")
            params.extend(val)
        else:
            if op not in ("=", ">", "<", "<=", ">=", "<>", "!=", "like"):
                op = "="
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)

    return where_clauses, params


@app.post("/ask")
async def ask(q: Query):
    schema = get_schema()

    try:
        plan = extract_query(q.question)

        if isinstance(plan, dict) and plan.get("error"):
            return {"sql": None, "result": f"AI error: {plan.get('error')}", "rows": [], "columns": []}

        # ---------------- YOY DETECTION ----------------
        yoy_filter = None
        for f in plan.get("filters", []) or []:
            if (
                isinstance(f, dict)
                and f.get("column") == "FinancialYear"
                and f.get("operator") == "in"
                and isinstance(f.get("value"), (list, tuple))
            ):
                yoy_filter = f
                break

        metric_key = detect_metric_from_text(q.question)
        metric_info = METRICS.get(metric_key) if metric_key else None

        # detect if user wants growth %
        want_growth = any(
            w in q.question.lower()
            for w in ["growth", "yoy", "increase", "rate", "%"]
        )

        if yoy_filter and metric_info and metric_info.get("expression"):
            years = list(yoy_filter.get("value"))
            if len(years) >= 2:

                metric_expr = metric_info.get("expression").strip()
                metric_alias = metric_key

                # build WHERE (excluding year IN)
                other_filters = [
                    f for f in plan.get("filters", [])
                    if not (f.get("column") == "FinancialYear" and f.get("operator") == "in")
                ]
                where_clauses, params = _filters_to_where(other_filters, schema)

                where_sql = ""
                if where_clauses:
                    where_sql = "WHERE " + " AND ".join(where_clauses)

                placeholders_years = ", ".join("?" for _ in years)
                params_for_query = list(params)
                params_for_query.extend(years)

                # ----------- Build YoY SQL (Difference only OR Difference + Growth) -----------
                growth_sql = ""
                if want_growth:
                    growth_sql = """
      , CASE
            WHEN LAG(Metric) OVER (ORDER BY [FinancialYear]) = 0 THEN NULL
            ELSE ((Metric - LAG(Metric) OVER (ORDER BY [FinancialYear])) * 100.0)
                  / LAG(Metric) OVER (ORDER BY [FinancialYear])
        END AS GrowthPct
                    """

                cte = f"""
WITH yearly AS (
  SELECT
    [FinancialYear],
    SUM({metric_expr}) AS Metric
  FROM {TABLE}
  {where_sql + (' AND ' if where_sql else 'WHERE ') } [FinancialYear] IN ({placeholders_years})
  GROUP BY [FinancialYear]
)
SELECT
  [FinancialYear] AS FinancialYear,
  Metric AS [{metric_alias}],
  LAG(Metric) OVER (ORDER BY [FinancialYear]) AS Prev{metric_alias},
  (Metric - LAG(Metric) OVER (ORDER BY [FinancialYear])) AS Difference
  {growth_sql}
FROM yearly
ORDER BY [FinancialYear];
""".strip()

                rows = run_sql(cte, params_for_query)

                # Fix NaN/Infinity
                def fix_float(v):
                    try:
                        if v is None:
                            return None
                        if isinstance(v, float):
                            if v != v or v == float("inf") or v == float("-inf"):
                                return None
                        return v
                    except:
                        return v

                normalized_rows = []
                for row in rows:
                    normalized_rows.append({k: fix_float(v) for k, v in row.items()})

                rows = normalized_rows

                # columns for UI
                columns = ["FinancialYear", metric_alias, f"Prev{metric_alias}", "Difference"]
                if want_growth:
                    columns.append("GrowthPct")

                return {
                    "sql": cte,
                    "result": f"{len(rows)} rows returned.",
                    "rows": rows,
                    "columns": columns
                }

        # ---------------- Normal Logic ----------------
        sql, params, columns = build_sql_from_plan(plan, TABLE, schema)
        rows = run_sql(sql, params)

        if not rows:
            return {"sql": sql, "result": "No data found", "rows": [], "columns": columns}

        normalized_rows = []
        for row in rows:
            new_row = {}
            for col in columns:
                new_row[col] = row.get(col) or row.get(col.lower()) or row.get(col.upper())
            normalized_rows.append(new_row)
        rows = normalized_rows

        if len(rows) == 1 and len(columns) == 1:
            val = rows[0].get(columns[0])
            if isinstance(val, (int, float)):
                val_fmt = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
            else:
                val_fmt = str(val)
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
                "rows": [], "columns": []
            }
        )
