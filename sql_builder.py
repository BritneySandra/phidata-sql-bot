# sql_builder.py

def build_sql_from_plan(plan, table_name, schema):
    """
    Convert the generic JSON plan into an executable SQL + parameter list.

    Supports:
    - Business metric expressions (e.g. REVAmount + WIPAmount)
    - Normal column aggregates
    - WHERE filters
    - GROUP BY
    - ORDER BY
    - LIMIT / TOP
    """

    selects = plan.get("select", [])
    filters = plan.get("filters", [])
    group_by = plan.get("group_by", [])
    order_by = plan.get("order_by", [])
    limit = plan.get("limit", None)

    sql_select_parts = []
    columns_out = []

    # ---------------------------------------------------
    # SELECT clause
    # ---------------------------------------------------
    for sel in selects:
        col = sel.get("column")
        expr = sel.get("expression")
        agg = sel.get("aggregation", "sum").upper()
        alias = sel.get("alias", "value")

        if expr:  
            # Business metric expression
            sql_select_parts.append(f"{agg}({expr}) AS [{alias}]")
            columns_out.append(alias)
        else:
            # Normal column
            sql_select_parts.append(f"{agg}([{col}]) AS [{alias}]")
            columns_out.append(alias)

    if not sql_select_parts:
        raise Exception("No valid select expressions in plan")

    select_sql = ", ".join(sql_select_parts)

    # ---------------------------------------------------
    # WHERE clause
    # ---------------------------------------------------
    where_clauses = []
    params = []

    for f in filters:
        col = f.get("column")
        op = f.get("operator", "=")
        val = f.get("value")

        if op.lower() == "in" and isinstance(val, list):
            placeholders = ",".join(["?"] * len(val))
            where_clauses.append(f"[{col}] IN ({placeholders})")
            params.extend(val)
        else:
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # ---------------------------------------------------
    # GROUP BY
    # ---------------------------------------------------
    if group_by:
        group_sql = " GROUP BY " + ", ".join([f"[{g}]" for g in group_by])
    else:
        group_sql = ""

    # ---------------------------------------------------
    # ORDER BY
    # ---------------------------------------------------
    if order_by:
        ob_parts = []
        for ob in order_by:
            col = ob.get("column")
            direction = ob.get("direction", "DESC").upper()
            ob_parts.append(f"[{col}] {direction}")
        order_sql = " ORDER BY " + ", ".join(ob_parts)
    else:
        order_sql = ""

    # ---------------------------------------------------
    # LIMIT / TOP
    # ---------------------------------------------------
    top_sql = ""
    if isinstance(limit, int) and limit > 0:
        top_sql = f"TOP {limit} "

    # ---------------------------------------------------
    # Final SQL
    # ---------------------------------------------------
    sql = f"""
    SELECT {top_sql}{select_sql}
    FROM {table_name}
    WHERE {where_sql}
    {group_sql}
    {order_sql}
    """.strip()

    return sql, params, columns_out
