import re

AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None

def build_sql_from_plan(plan, table, schema):
    selects = plan.get("select", [])
    filters = plan.get("filters", [])
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    sql_select_parts = []
    metric_alias = None

    # include group_by columns first (ensure uniqueness & schema-valid)
    seen_cols = set()
    for col in group_by:
        if col in schema and col not in seen_cols:
            sql_select_parts.append(f"[{col}]")
            seen_cols.add(col)

    # add selects, preventing nested aggs and duplicates
    for sel in selects:
        col = sel.get("column")
        expr = sel.get("expression")
        agg = sel.get("aggregation")
        alias = sel.get("alias") or (col if col else "value")

        # normalize aggregation
        if agg is not None and (str(agg).lower() == "none" or str(agg).lower() == "null"):
            agg = None

        # track metric alias for ordering fallback
        if agg:
            metric_alias = alias

        # if expression provided
        if expr and isinstance(expr, str) and expr.strip():
            if is_aggregate_expression(expr):
                # expression already aggregate -> don't wrap
                sql_select_parts.append(f"{expr} AS [{alias}]")
            else:
                if agg:
                    sql_select_parts.append(f"{agg}({expr}) AS [{alias}]")
                else:
                    sql_select_parts.append(f"{expr} AS [{alias}]")
            continue

        # if column provided
        if col:
            if col in seen_cols:
                # skip duplicate column selection
                continue
            if agg:
                sql_select_parts.append(f"{agg}([{col}]) AS [{alias}]")
            else:
                sql_select_parts.append(f"[{col}] AS [{alias}]")
            seen_cols.add(col)
            continue

    if not sql_select_parts:
        raise Exception("No valid SELECT expressions in plan")

    sql = f"SELECT {', '.join(sql_select_parts)} FROM {table}"

    # WHERE
    where_clauses = []
    params = []
    for flt in filters:
        col = flt.get("column")
        op = flt.get("operator", "=")
        val = flt.get("value")
        if col and col in schema:
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # GROUP BY
    if group_by:
        sql += " GROUP BY " + ", ".join(f"[{c}]" for c in group_by)

    # ORDER BY
    if order_by:
        sql += " ORDER BY " + ", ".join(
            f"[{ob['column']}] {ob.get('direction','DESC')}" for ob in order_by
        )
    elif limit and metric_alias:
        sql += f" ORDER BY [{metric_alias}] DESC"

    # LIMIT (TOP N)
    if limit:
        sql = f"SELECT TOP {limit} " + sql[7:]

    # output columns for UI
    columns = []
    for c in group_by:
        columns.append(c)
    for s in selects:
        a = s.get("alias")
        if a:
            columns.append(a)
    # dedupe preserve order
    seen = set()
    cols_ordered = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            cols_ordered.append(c)
    return sql, params, cols_ordered
