# sql_builder.py - safe SQL builder (enhanced ordering, alias handling, dimension support)
import re

AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return bool(AGG_RE.match(expr.strip()))

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

    # -------------------------------------------------------------
    # 1. Ensure group-by columns appear in SELECT
    # -------------------------------------------------------------
    for col in group_by:
        if col in schema:
            if col not in seen_cols:
                sql_select_parts.append(f"[{col}] AS [{col}]")
                seen_cols.add(col)

    # -------------------------------------------------------------
    # 2. Build SELECT expressions
    # -------------------------------------------------------------
    for sel in selects:
        if not isinstance(sel, dict):
            continue

        col = sel.get("column")
        expr = sel.get("expression")
        agg = sel.get("aggregation")
        alias = sel.get("alias") or col or "value"

        if agg and str(agg).lower() in ("none", "null"):
            agg = None

        # Remember metric alias for ORDER BY
        if agg or (expr and is_aggregate_expression(expr)):
            metric_alias = alias

        # -------------------------
        # Case 1: Expression present
        # -------------------------
        if expr and isinstance(expr, str) and expr.strip():
            expr_clean = expr.strip()

            # Expression already contains an aggregate e.g. SUM(), AVG(), COUNT()
            if is_aggregate_expression(expr_clean):
                sql_select_parts.append(f"{expr_clean} AS [{alias}]")
            else:
                # Wrap expression
                if agg:
                    sql_select_parts.append(f"{agg}({expr_clean}) AS [{alias}]")
                else:
                    sql_select_parts.append(f"{expr_clean} AS [{alias}]")
            continue

        # -------------------------
        # Case 2: Column present
        # -------------------------
        if col:
            if col not in seen_cols:
                if agg:
                    sql_select_parts.append(f"{agg}([{col}]) AS [{alias}]")
                else:
                    sql_select_parts.append(f"[{col}] AS [{alias}]")
                seen_cols.add(col)
            continue

    # -------------------------------------------------------------
    # Safety: no SELECT columns → fail
    # -------------------------------------------------------------
    if not sql_select_parts:
        raise Exception("No valid SELECT expressions in plan")

    # -------------------------------------------------------------
    # Base query
    # -------------------------------------------------------------
    sql = f"SELECT {', '.join(sql_select_parts)} FROM {table}"

    # -------------------------------------------------------------
    # WHERE
    # -------------------------------------------------------------
    where_clauses = []
    params = []

    for flt in filters:
        if not isinstance(flt, dict):
            continue

        col = flt.get("column")
        op = (flt.get("operator") or "=").lower()
        val = flt.get("value")

        if not col or col not in schema:
            # skip unknown columns
            continue

        if op == "in" and isinstance(val, (list, tuple)):
            placeholders = ", ".join("?" for _ in val)
            where_clauses.append(f"[{col}] IN ({placeholders})")
            params.extend(val)
        else:
            # default equality / other operator applied as-is
            # Sanitize operator to common allowed ones
            if op not in ("=", ">", "<", "<=", ">=", "<>", "!=", "like"):
                op = "="
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # -------------------------------------------------------------
    # GROUP BY
    # -------------------------------------------------------------
    if group_by:
        gb_valid = [c for c in group_by if c in schema]
        if gb_valid:
            sql += " GROUP BY " + ", ".join(f"[{c}]" for c in gb_valid)

    # -------------------------------------------------------------
    # ORDER BY — improved alias handling
    # -------------------------------------------------------------
    if order_by:
        ob_parts = []

        for ob in order_by:
            if not isinstance(ob, dict):
                continue

            col = ob.get("column")
            direction = ob.get("direction", "DESC")

            if not col:
                continue

            # If alias exists, use alias
            alias_list = [s.get("alias") for s in selects if isinstance(s, dict)]
            if col in alias_list:
                ob_parts.append(f"[{col}] {direction}")
            else:
                # Otherwise fallback to raw column
                if col in schema:
                    ob_parts.append(f"[{col}] {direction}")
                else:
                    # As last resort treat as alias
                    ob_parts.append(f"[{col}] {direction}")

        if ob_parts:
            sql += " ORDER BY " + ", ".join(ob_parts)

    # -------------------------------------------------------------
    # Fallback ordering when LIMIT is used and no ORDER BY
    # -------------------------------------------------------------
    # LIMIT / TOP
    # -------------------------------------------------------------
    if limit:
        try:
            n = int(limit)
            sql = sql.replace("SELECT ", f"SELECT TOP {n} ", 1)
        except:
            pass

    # -------------------------------------------------------------
    # Output column order for UI
    # -------------------------------------------------------------
    columns = []

    # group-by columns first
    for c in group_by:
        columns.append(c)

    # then metric/dimension aliases
    for s in selects:
        if isinstance(s, dict):
            if s.get("alias"):
                columns.append(s["alias"])

    # remove duplicates
    seen = set()
    cols_ordered = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            cols_ordered.append(c)

    return sql, params, cols_ordered
