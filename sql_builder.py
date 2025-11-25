# sql_builder.py - safe SQL builder
import re

AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return AGG_RE.match(expr.strip()) is not None

def build_sql_from_plan(plan, table, schema):
    """
    Returns: sql, params, columns_ordered
    plan must be a dict with select/filters/group_by/order_by/limit
    """
    if not isinstance(plan, dict):
        raise Exception("AI returned non-dict plan")

    selects = plan.get("select", []) or []
    filters = plan.get("filters", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    sql_select_parts = []
    metric_alias = None

    seen_cols = set()
    for col in group_by:
        if col in schema and col not in seen_cols:
            sql_select_parts.append(f"[{col}]")
            seen_cols.add(col)

    for sel in selects:
        if not isinstance(sel, dict):
            continue
        col = sel.get("column")
        expr = sel.get("expression")
        agg = sel.get("aggregation")
        alias = sel.get("alias") or (col if col else "value")

        if agg is not None and (str(agg).lower() in ("none","null")):
            agg = None

        if agg:
            metric_alias = alias

        # expression provided
        if expr and isinstance(expr, str) and expr.strip():
            if is_aggregate_expression(expr):
                sql_select_parts.append(f"{expr} AS [{alias}]")
            else:
                if agg:
                    sql_select_parts.append(f"{agg}({expr}) AS [{alias}]")
                else:
                    sql_select_parts.append(f"{expr} AS [{alias}]")
            continue

        # column provided
        if col:
            if col in seen_cols:
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

    where_clauses = []
    params = []
    for flt in filters:
        if not isinstance(flt, dict):
            continue
        col = flt.get("column")
        op = flt.get("operator", "=") or "="
        val = flt.get("value")
        if col and col in schema:
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    if group_by:
        # ensure group_by includes only valid schema columns
        gb_valid = [c for c in group_by if c in schema]
        if gb_valid:
            sql += " GROUP BY " + ", ".join(f"[{c}]" for c in gb_valid)

    # ORDER BY
    if order_by:
        ob_parts = []
        for ob in order_by:
            if not isinstance(ob, dict):
                continue
            col = ob.get("column")
            direction = ob.get("direction", "DESC")
            if col:
                ob_parts.append(f"[{col}] {direction}")
        if ob_parts:
            sql += " ORDER BY " + ", ".join(ob_parts)
    elif limit and metric_alias:
        sql += f" ORDER BY [{metric_alias}] DESC"

    # LIMIT (SQL Server uses TOP)
    if limit:
        try:
            n = int(limit)
            sql = f"SELECT TOP {n} " + sql[len("SELECT "):]
        except Exception:
            # ignore invalid limit
            pass

    # build output columns order for UI
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
