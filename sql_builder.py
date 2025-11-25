import re

AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr):
    return isinstance(expr, str) and AGG_RE.match(expr.strip())

def build_sql_from_plan(plan, table, schema):
    if "error" in plan:
        raise Exception(f"Invalid plan returned from LLM: {plan}")

    selects = plan["select"]
    filters = plan["filters"]
    group_by = plan["group_by"]
    order_by = plan["order_by"]
    limit = plan["limit"]

    sql_parts = []
    metric_alias = None
    seen = set()

    # Group by first
    for col in group_by:
        if col in schema and col not in seen:
            sql_parts.append(f"[{col}]")
            seen.add(col)

    # SELECTS
    for sel in selects:
        col = sel.get("column")
        expr = sel.get("expression")
        agg = sel.get("aggregation")
        alias = sel.get("alias") or (col or "value")

        if agg in ["none", "null", None]:
            agg = None

        if expr and is_aggregate_expression(expr):
            sql_parts.append(f"{expr} AS [{alias}]")
            continue

        if expr:
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
            continue

    sql = f"SELECT {', '.join(sql_parts)} FROM {table}"

    # WHERE
    where_parts = []
    params = []
    for flt in filters:
        col = flt["column"]
        op = flt.get("operator", "=")
        val = flt["value"]

        if col in schema:
            where_parts.append(f"[{col}] {op} ?")
            params.append(val)

    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    # GROUP BY
    if group_by:
        sql += " GROUP BY " + ", ".join(f"[{c}]" for c in group_by)

    # ORDER BY
    if order_by:
        sql += " ORDER BY " + ", ".join(
            f"[{o['column']}] {o.get('direction','DESC')}" for o in order_by
        )

    # LIMIT
    if limit:
        sql = "SELECT TOP " + str(limit) + " " + sql[7:]

    # Build columns list
    columns = []
    for c in group_by:
        columns.append(c)
    for s in selects:
        columns.append(s.get("alias"))

    # unique
    final_cols = []
    seen = set()
    for c in columns:
        if c and c not in seen:
            seen.add(c)
            final_cols.append(c)

    return sql, params, final_cols
