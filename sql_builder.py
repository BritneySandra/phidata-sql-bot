# sql_builder.py - enhanced SQL builder with MOM / QOQ / YOY CTE + LAG support
import re

AGG_RE = re.compile(r"^\s*(sum|avg|count|min|max)\s*\(", re.IGNORECASE)

def is_aggregate_expression(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    return bool(AGG_RE.match(expr.strip()))

def _wrap_agg(agg, expr):
    if not agg:
        return expr
    return f"{agg}({expr})"

def _find_filters(plan_filters, col_name):
    """Return list of filters matching column name"""
    out = []
    for f in plan_filters or []:
        if not isinstance(f, dict):
            continue
        if f.get("column") == col_name:
            out.append(f)
    return out

def _filter_values_as_params(flt):
    """Return (sql_snippet, params_list) for a single filter object"""
    op = (flt.get("operator") or "=").lower()
    val = flt.get("value")
    if op == "in" and isinstance(val, (list, tuple)):
        placeholders = ", ".join("?" for _ in val)
        return f"IN ({placeholders})", list(val)
    else:
        # treat as single equality/other operator
        if op not in ("=", ">", "<", "<=", ">=", "<>", "!=", "like"):
            op = "="
        return f"{op} ?", [val]

def build_sql_from_plan(plan, table, schema):
    """
    Returns: sql, params, columns_ordered

    Enhanced builder:
    - If plan groups by FinancialYear + FinancialMonth OR FinancialYear + FinancialQuarter
      and one metric select exists, build a CTE that aggregates Metric then use LAG() to compute PrevMetric, Difference, GrowthPct.
    - Support partitioning LAG by non-time group-by dimensions (e.g., TransportMode).
    - Fallback to previous generic builder if no special time grouping detected.
    """
    if not isinstance(plan, dict):
        raise Exception("AI returned non-dict plan")

    selects = plan.get("select", []) or []
    filters = plan.get("filters", []) or []
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    # normalize simple shapes
    # find metric select: any select that has expression or aggregation (non-dimension)
    metric_select = None
    metric_alias = None
    metric_expr_raw = None
    metric_agg = None

    # prefer selects that appear not to be direct group-by columns
    for s in selects:
        if not isinstance(s, dict):
            continue
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or (col if col else "value")
        # treat group-by columns as dims not metrics
        if col and col in group_by and not expr and not agg:
            continue
        # Found candidate metric
        metric_select = s
        metric_alias = alias
        metric_expr_raw = expr if expr else (f"[{col}]" if col else None)
        metric_agg = agg
        break

    # helper to build a safe aggregate expression string
    def build_metric_expression_str(select):
        if not select:
            return None
        col = select.get("column")
        expr = select.get("expression")
        agg = select.get("aggregation")
        alias = select.get("alias") or (col if col else "value")
        if expr and isinstance(expr, str) and expr.strip():
            expr_clean = expr.strip()
            # If expression is already an aggregate, use as-is
            if is_aggregate_expression(expr_clean):
                return expr_clean, alias
            # else wrap with aggregation if provided
            if agg:
                return f"{agg}({expr_clean})", alias
            return expr_clean, alias
        if col:
            if agg:
                return f"{agg}([{col}])", alias
            return f"[{col}]", alias
        return None, alias

    metric_expr_str, metric_alias = build_metric_expression_str(metric_select)

    # Detect time grouping patterns
    gb_set = set(group_by)
    has_fy = "FinancialYear" in gb_set or any((f.get("column") == "FinancialYear") for f in filters)
    has_fm = "FinancialMonth" in gb_set or any((f.get("column") == "FinancialMonth") for f in filters)
    has_fq = "FinancialQuarter" in gb_set or any((f.get("column") == "FinancialQuarter") for f in filters)

    # Determine partition dims (group_by excluding time columns)
    partition_dims = [c for c in group_by if c not in ("FinancialYear", "FinancialMonth", "FinancialQuarter")]

    # If we have a time-grouping (FY+FM or FY+FQ or FY with IN) AND a metric -> build special CTE with LAG
    if metric_expr_str and ((has_fy and has_fm) or (has_fy and has_fq) or (has_fy and (any(f.get("operator") == "in" and f.get("column") == "FinancialYear" for f in filters)))):
        # choose time grain
        if has_fy and has_fm:
            time_cols = ["FinancialYear", "FinancialMonth"]
            order_cols = ["[FinancialYear]", "[FinancialMonth]"]
        elif has_fy and has_fq:
            time_cols = ["FinancialYear", "FinancialQuarter"]
            order_cols = ["[FinancialYear]", "[FinancialQuarter]"]
        else:
            # YoY fallback: only FinancialYear
            time_cols = ["FinancialYear"]
            order_cols = ["[FinancialYear]"]

        # Build WHERE clause for CTE from filters but keep only filters that don't conflict with group_by IN logic.
        where_clauses = []
        params = []
        # We will gather IN values for time cols to ensure the CTE is restricted to required years/months/quarters
        for flt in filters:
            if not isinstance(flt, dict):
                continue
            col = flt.get("column")
            op = (flt.get("operator") or "=").lower()
            val = flt.get("value")
            # If filter is on time cols and operator 'in' we will include it as usual
            if col in (time_cols):
                if op == "in" and isinstance(val, (list, tuple)):
                    placeholders = ", ".join("?" for _ in val)
                    where_clauses.append(f"[{col}] IN ({placeholders})")
                    params.extend(list(val))
                else:
                    # equality/single
                    where_clauses.append(f"[{col}] {op} ?")
                    params.append(val)
            else:
                # non-time filter included as-is (if it's a known schema column)
                if col in (schema or {}):
                    if op == "in" and isinstance(val, (list, tuple)):
                        placeholders = ", ".join("?" for _ in val)
                        where_clauses.append(f"[{col}] IN ({placeholders})")
                        params.extend(list(val))
                    else:
                        if op not in ("=", ">", "<", "<=", ">=", "<>", "!=", "like"):
                            op = "="
                        where_clauses.append(f"[{col}] {op} ?")
                        params.append(val)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Build GROUP BY for CTE: include time_cols + partition_dims
        cte_group_cols = time_cols + partition_dims
        cte_group_clause = ", ".join(f"[{c}]" for c in cte_group_cols) if cte_group_cols else ""

        # Metric alias used inside CTE: call it Metric
        metric_sql_expr = metric_expr_str

        cte_select_cols = ", ".join(f"[{c}]" for c in cte_group_cols) if cte_group_cols else ""
        if cte_select_cols:
            cte_select_cols = cte_select_cols + ", "

        cte = f"WITH aggregated AS (\n  SELECT\n    {cte_select_cols}{metric_sql_expr} AS Metric\n  FROM {table}\n  {where_sql}\n"
        if cte_group_clause:
            cte += f"  GROUP BY {cte_group_clause}\n"
        cte += ")\n"

        # Build partition clause for LAG (if partition_dims exist)
        partition_sql = ""
        if partition_dims:
            partition_sql = "PARTITION BY " + ", ".join(f"[{c}]" for c in partition_dims) + " "

        # Compose LAG expression with ORDER BY appropriate time order
        order_by_sql = ", ".join(order_cols)
        lag_expr = f"LAG(Metric) OVER ({partition_sql}ORDER BY {order_by_sql})"

        # Final select: include partition dims and time cols in the select ordering same as group_by preference
        final_select_cols = []
        # Maintain original group_by ordering where possible: present group_by columns first (keeps dimension then time)
        for g in group_by:
            if g in partition_dims + time_cols:
                final_select_cols.append(f"[{g}] AS [{g}]")
        # If that yields nothing (rare), fallback to time cols then dims
        if not final_select_cols:
            for c in time_cols + partition_dims:
                final_select_cols.append(f"[{c}] AS [{c}]")

        # metric column named by alias from the plan (fallback to metric_alias computed earlier)
        display_alias = metric_alias or "metric"
        final_select_cols.append(f"Metric AS [{display_alias}]")
        final_select_cols.append(f"{lag_expr} AS Prev{display_alias}")
        final_select_cols.append(f"(Metric - {lag_expr}) AS Difference")
        final_select_cols.append(
            "CASE WHEN {lag} = 0 THEN NULL ELSE ((Metric - {lag}) * 100.0) / {lag} END AS GrowthPct".format(
                lag=lag_expr
            )
        )

        final_select_sql = ",\n  ".join(final_select_cols)

        sql = f"{cte}SELECT\n  {final_select_sql}\nFROM aggregated\nORDER BY {order_by_sql};"

        # Build returned columns list for UI
        columns = []
        # keep same ordering as final_select_cols but use simple names
        for g in group_by:
            if g in partition_dims + time_cols and g not in columns:
                columns.append(g)
        # ensure time cols present
        for tc in time_cols:
            if tc not in columns:
                columns.append(tc)
        # metric + derived
        columns += [display_alias, f"Prev{display_alias}", "Difference", "GrowthPct"]

        return sql, params, columns

    # ---------- FALLBACK GENERIC BUILDER (original behavior) ----------
    sql_select_parts = []
    seen_cols = set()

    # Ensure group-by columns appear in SELECT
    for col in group_by:
        if col in schema:
            if col not in seen_cols:
                sql_select_parts.append(f"[{col}] AS [{col}]")
                seen_cols.add(col)

    # Build SELECT expressions
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
        # (not strictly necessary here)

        if expr and isinstance(expr, str) and expr.strip():
            expr_clean = expr.strip()
            if is_aggregate_expression(expr_clean):
                sql_select_parts.append(f"{expr_clean} AS [{alias}]")
            else:
                if agg:
                    sql_select_parts.append(f"{agg}({expr_clean}) AS [{alias}]")
                else:
                    sql_select_parts.append(f"{expr_clean} AS [{alias}]")
            continue

        if col:
            if col not in seen_cols:
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
        if not isinstance(flt, dict):
            continue
        col = flt.get("column")
        op = (flt.get("operator") or "=").lower()
        val = flt.get("value")
        if not col or col not in schema:
            continue
        if op == "in" and isinstance(val, (list, tuple)):
            placeholders = ", ".join("?" for _ in val)
            where_clauses.append(f"[{col}] IN ({placeholders})")
            params.extend(val)
        else:
            if op not in ("=", ">", "<", "<=", ">=", "<>", "!=", "like"):
                op = "="
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # GROUP BY
    if group_by:
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
            if not col:
                continue
            alias_list = [s.get("alias") for s in selects if isinstance(s, dict)]
            if col in alias_list:
                ob_parts.append(f"[{col}] {direction}")
            else:
                if col in schema:
                    ob_parts.append(f"[{col}] {direction}")
                else:
                    ob_parts.append(f"[{col}] {direction}")
        if ob_parts:
            sql += " ORDER BY " + ", ".join(ob_parts)

    # LIMIT / TOP
    if limit:
        try:
            n = int(limit)
            sql = sql.replace("SELECT ", f"SELECT TOP {n} ", 1)
        except:
            pass

    # Build columns order for UI
    columns = []
    for c in group_by:
        columns.append(c)
    for s in selects:
        if isinstance(s, dict):
            if s.get("alias"):
                columns.append(s["alias"])
    # dedupe preserve order
    seen = set()
    cols_ordered = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            cols_ordered.append(c)

    return sql, params, cols_ordered
