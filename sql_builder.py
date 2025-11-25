# sql_builder.py — safer, defensive SQL builder

import re

AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None


def build_sql_from_plan(plan, table, schema):
    """
    Build SQL from normalized plan.
    Returns (sql, params, columns) where params are in order for '?' placeholders.
    """
    selects = plan.get("select", []) or []
    filters = plan.get("filters", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    sql_select_parts = []
    metric_alias = None

    # include group_by columns first (ensure uniqueness & schema-valid)
    seen_cols = set()
    valid_group_by = []
    for col in group_by:
        if isinstance(col, str) and col in schema and col not in seen_cols:
            sql_select_parts.append(f"[{col}]")
            seen_cols.add(col)
            valid_group_by.append(col)
    group_by = valid_group_by  # use validated group_by

    # add selects, preventing nested aggs and duplicates
    for sel in selects:
        if not isinstance(sel, dict):
            # skip malformed selects
            continue

        col = sel.get("column")
        expr = sel.get("expression")
        agg = sel.get("aggregation")
        alias = sel.get("alias") or (col if col else None)

        # normalize aggregation (guard against strings like "None"/"null")
        if agg is not None and isinstance(agg, str) and agg.lower() in ("none", "null"):
            agg = None

        # if alias missing, generate safe alias (avoid None)
        if not alias:
            # create alias from column or expression
            if col:
                alias = col
            else:
                alias = "value"

        # Only set metric_alias when we produce an aggregated metric column (not when we add simple group_by)
        produced_expr = None
        if expr and isinstance(expr, str) and expr.strip():
            # If expression already contains an aggregate, don't wrap it
            if is_aggregate_expression(expr):
                produced_expr = f"{expr} AS [{alias}]"
            else:
                if agg:
                    produced_expr = f"{agg}({expr}) AS [{alias}]"
                else:
                    produced_expr = f"{expr} AS [{alias}]"

        elif col:
            if col in seen_cols:
                # skip duplicate column selection
                continue
            if agg:
                produced_expr = f"{agg}([{col}]) AS [{alias}]"
            else:
                produced_expr = f"[{col}] AS [{alias}]"
            seen_cols.add(col)

        if produced_expr:
            sql_select_parts.append(produced_expr)
            # If this select is an aggregation, set metric_alias for ordering fallback
            if agg:
                metric_alias = alias

    if not sql_select_parts:
        raise Exception("No valid SELECT expressions in plan")

    sql = f"SELECT {', '.join(sql_select_parts)} FROM {table}"

    # WHERE - build clauses only for columns existing in schema OR allow column aliases
    where_clauses = []
    params = []
    for flt in filters:
        if not isinstance(flt, dict):
            continue
        col = flt.get("column")
        op = flt.get("operator", "=") or "="
        val = flt.get("value")

        # prefer to only param on known column names; but allow if alias used (we still parameterize)
        if not col:
            continue

        # sanitize operator (simple allowlist)
        op = str(op).strip().upper()
        if op not in ("=", "<>", "!=", ">", "<", ">=", "<=", "IN", "LIKE"):
            op = "="

        if op == "IN" and isinstance(val, (list, tuple)):
            # create placeholders for IN clause
            placeholders = ", ".join("?" for _ in val)
            where_clauses.append(f"[{col}] IN ({placeholders})")
            params.extend(val)
        else:
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # GROUP BY - use validated group_by
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
            # allow ordering by alias or column (we don't require schema check here)
            order_parts.append(f"[{col}] {direction}")

    if order_parts:
        sql += " ORDER BY " + ", ".join(order_parts)
    elif limit and metric_alias:
        # fallback ordering by metric alias if available
        sql += f" ORDER BY [{metric_alias}] DESC"

    # LIMIT (TOP N) — apply TOP if limit present
    if limit:
        # guard limit type
        try:
            lim = int(limit)
            if lim > 0:
                sql = f"SELECT TOP {lim} " + sql[7:]
        except Exception:
            # ignore invalid limit
            pass

    # output columns for UI (preserve order and dedupe)
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
