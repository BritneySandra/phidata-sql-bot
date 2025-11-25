# sql_builder.py — defensive SQL builder (Option C-compatible)

import re

AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None

def build_sql_from_plan(plan, table, schema):
    """
    Build SQL from normalized plan (output of agent.normalize_plan).
    Returns tuple (sql, params, columns_list).
    """
    selects = plan.get("select", []) or []
    filters = plan.get("filters", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    sql_select_parts = []
    metric_alias = None

    # Validate and include group_by cols first
    seen_cols = set()
    valid_group_by = []
    for col in group_by:
        if isinstance(col, str) and col in schema and col not in seen_cols:
            sql_select_parts.append(f"[{col}]")
            seen_cols.add(col)
            valid_group_by.append(col)
    group_by = valid_group_by

    # Build SELECT items
    for sel in selects:
        if not isinstance(sel, dict):
            continue
        col = sel.get("column")
        expr = sel.get("expression")
        agg = sel.get("aggregation")
        alias = sel.get("alias") or (col if col else None)

        if agg is not None and isinstance(agg, str) and agg.lower() in ("none", "null"):
            agg = None

        if not alias:
            alias = col if col else ("expr" if expr else "value")

        produced = None
        # If expression exists:
        if isinstance(expr, str) and expr.strip():
            # If expr already contains aggregate -> use as-is
            if is_aggregate_expression(expr):
                produced = f"{expr} AS [{alias}]"
            else:
                if agg:
                    produced = f"{agg}({expr}) AS [{alias}]"
                else:
                    produced = f"{expr} AS [{alias}]"
        elif col:
            if col in seen_cols:
                continue
            if agg:
                produced = f"{agg}([{col}]) AS [{alias}]"
            else:
                produced = f"[{col}] AS [{alias}]"
            seen_cols.add(col)

        if produced:
            sql_select_parts.append(produced)
            if agg:
                metric_alias = alias

    if not sql_select_parts:
        raise Exception("No valid SELECT expressions in plan")

    sql = f"SELECT {', '.join(sql_select_parts)} FROM {table}"

    # WHERE clause + params
    where_clauses = []
    params = []
    for flt in filters:
        if not isinstance(flt, dict):
            continue
        col = flt.get("column")
        op = flt.get("operator", "=") or "="
        val = flt.get("value")

        if not col:
            continue

        op = str(op).strip().upper()
        if op not in ("=", "<>", "!=", ">", "<", ">=", "<=", "IN", "LIKE"):
            op = "="

        if op == "IN" and isinstance(val, (list, tuple)):
            placeholders = ", ".join("?" for _ in val)
            where_clauses.append(f"[{col}] IN ({placeholders})")
            params.extend(val)
        else:
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # GROUP BY
    if group_by:
        sql += " GROUP BY " + ", ".join(f"[{c}]" for c in group_by)

    # ORDER BY
    order_parts = []
    if isinstance(order_by, list):
        for ob in order_by:
            if not isinstance(ob, dict):
                continue
            col = ob.get("column")
            direction = ob.get("direction", "DESC") or "DESC"
            if not col:
                continue
            direction = str(direction).upper()
            if direction not in ("ASC", "DESC"):
                direction = "DESC"
            order_parts.append(f"[{col}] {direction}")

    if order_parts:
        sql += " ORDER BY " + ", ".join(order_parts)
    elif limit and metric_alias:
        sql += f" ORDER BY [{metric_alias}] DESC"

    # LIMIT (TOP N)
    if limit:
        try:
            lim = int(limit)
            if lim > 0:
                sql = f"SELECT TOP {lim} " + sql[7:]
        except Exception:
            pass

    # Columns for UI: group_by cols then aliases from selects (dedup)
    columns = []
    for c in group_by:
        columns.append(c)
    for s in selects:
        if isinstance(s, dict):
            a = s.get("alias")
            if a:
                columns.append(a)
    seen = set()
    cols_ordered = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            cols_ordered.append(c)

    return sql, params, cols_ordered
