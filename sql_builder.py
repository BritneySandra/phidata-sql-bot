import re

def build_sql_from_plan(plan, table, schema):
    selects = plan.get("select", [])
    filters = plan.get("filters", [])
    group_by = plan.get("group_by", [])
    order_by = plan.get("order_by", [])
    limit = plan.get("limit")

    sql_select_parts = []

    # --- 1. SELECT grouping dimensions ---
    for col in group_by:
        if col in schema:
            sql_select_parts.append(f"[{col}]")

    # --- 2. SELECT metrics ---
    for sel in selects:
        col = sel.get("column")
        expr = sel.get("expression")
        agg = sel.get("aggregation", "sum")
        alias = sel.get("alias", "value")

        if expr:
            sql_select_parts.append(f"{agg}({expr}) AS [{alias}]")
        elif col:
            sql_select_parts.append(f"{agg}([{col}]) AS [{alias}]")

    if not sql_select_parts:
        raise Exception("No valid select expressions in plan")

    sql = f"SELECT {', '.join(sql_select_parts)} FROM {table}"

    # --- 3. WHERE clause ---
    params = []
    where_clauses = []

    for flt in filters:
        col = flt.get("column")
        op = flt.get("operator", "=")
        val = flt.get("value")

        if col in schema:
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # --- 4. GROUP BY ---
    if group_by:
        sql += " GROUP BY " + ", ".join(f"[{c}]" for c in group_by)

    # --- 5. ORDER BY ---
    if order_by:
        sql += " ORDER BY " + ", ".join(
            f"[{ob['column']}] {ob.get('direction','DESC')}"
            for ob in order_by
        )

    # --- 6. LIMIT ---
    if limit:
        sql = f"SELECT TOP {limit} " + sql[7:]

    columns = group_by + [sel.get("alias") for sel in selects]

    return sql, params, columns
