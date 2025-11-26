# sql_builder.py - safe SQL builder (enhanced)
import re

AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return bool(AGG_RE.match(expr.strip()))

def build_sql_from_plan(plan, table, schema):
    if not isinstance(plan, dict):
        raise Exception("AI returned non-dict plan")

    selects = plan.get("select", [])
    filters = plan.get("filters", [])
    group_by = plan.get("group_by", [])
    order_by = plan.get("order_by", [])
    limit = plan.get("limit")

    sql_parts = []
    seen = set()
    metric_alias = None

    for g in group_by:
        if g in schema:
            sql_parts.append(f"[{g}] AS [{g}]")
            seen.add(g)

    for sel in selects:
        if not isinstance(sel, dict):
            continue

        col = sel.get("column")
        expr = sel.get("expression")
        agg = sel.get("aggregation")
        alias = sel.get("alias") or col or "value"

        if agg in ("none", "null"):
            agg = None

        if agg or (expr and is_aggregate_expression(expr)):
            metric_alias = alias

        if expr:
            if is_aggregate_expression(expr):
                sql_parts.append(f"{expr} AS [{alias}]")
            else:
                if agg:
                    sql_parts.append(f"{agg}({expr}) AS [{alias}]")
                else:
                    sql_parts.append(f"{expr} AS [{alias}]")
            continue

        if col:
            if col not in seen:
                if agg:
                    sql_parts.append(f"{agg}([{col}]) AS [{alias}]")
                else:
                    sql_parts.append(f"[{col}] AS [{alias}]")
                seen.add(col)

    if not sql_parts:
        raise Exception("No SELECT expressions")

    sql = f"SELECT {', '.join(sql_parts)} FROM {table}"

    where = []
    params = []

    for f in filters:
        col = f.get("column")
        op = f.get("operator", "=")
        val = f.get("value")
        if col in schema:
            where.append(f"[{col}] {op} ?")
            params.append(val)

    if where:
        sql += " WHERE " + " AND ".join(where)

    if group_by:
        grouped = [c for c in group_by if c in schema]
        if grouped:
            sql += " GROUP BY " + ", ".join(f"[{c}]" for c in grouped)

    if order_by:
        obs = []
        aliases = [s.get("alias") for s in selects if isinstance(s, dict)]
        for ob in order_by:
            col = ob.get("column")
            direction = ob.get("direction", "DESC")
            if col in aliases:
                obs.append(f"[{col}] {direction}")
            elif col in schema:
                obs.append(f"[{col}] {direction}")
        if obs:
            sql += " ORDER BY " + ", ".join(obs)

    if limit:
        try:
            n = int(limit)
            sql = "SELECT TOP " + str(n) + " " + sql[len("SELECT "):]
        except:
            pass

    ordered_cols = []
    for g in group_by:
        ordered_cols.append(g)
    for s in selects:
        if s.get("alias"):
            ordered_cols.append(s["alias"])

    final_cols = []
    seen = set()
    for c in ordered_cols:
        if c not in seen:
            seen.add(c)
            final_cols.append(c)

    return sql, params, final_cols
