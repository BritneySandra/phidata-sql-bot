# ===========================
# sql_builder.py (FINAL)
# ===========================
import re

AGG_RE = re.compile(r"(sum|avg|count|min|max)\(", re.IGNORECASE)

def is_aggregate_expression(expr):
    if not expr or not isinstance(expr,str):
        return False
    return bool(AGG_RE.search(expr))

def build_sql_from_plan(plan, table, schema):
    selects = plan.get("select", [])
    filters = plan.get("filters", [])
    group_by = plan.get("group_by", []) or []
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    sql_select_parts = []
    metric_alias = None

    seen = set()

    # group_by first
    for c in group_by:
        if c in schema and c not in seen:
            sql_select_parts.append(f"[{c}]")
            seen.add(c)

    # add selects
    for s in selects:
        col = s.get("column")
        expr = s.get("expression")
        agg = s.get("aggregation")
        alias = s.get("alias") or (col if col else "value")

        if agg in ("none","null","None",None):
            agg = None

        if agg:
            metric_alias = alias

        # expression
        if expr:
            if is_aggregate_expression(expr):
                sql_select_parts.append(f"{expr} AS [{alias}]")
            else:
                if agg:
                    sql_select_parts.append(f"{agg}({expr}) AS [{alias}]")
                else:
                    sql_select_parts.append(f"{expr} AS [{alias}]")
            continue

        # column
        if col and col not in seen:
            if agg:
                sql_select_parts.append(f"{agg}([{col}]) AS [{alias}]")
            else:
                sql_select_parts.append(f"[{col}] AS [{alias}]")
            seen.add(col)

    sql = f"SELECT {', '.join(sql_select_parts)} FROM {table}"

    # WHERE
    params = []
    where_clauses = []

    for f in filters:
        col = f.get("column")
        op = f.get("operator", "=")
        val = f.get("value")

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
            f"[{o['column']}] {o.get('direction','DESC')}"
            for o in order_by
        )
    elif limit and metric_alias:
        sql += f" ORDER BY [{metric_alias}] DESC"

    # LIMIT
    if limit:
        sql = f"SELECT TOP {limit} " + sql[7:]

    # output columns
    cols = []
    cols.extend(group_by)
    cols.extend([s.get("alias") for s in selects if s.get("alias")])

    final_cols = []
    seen = set()
    for c in cols:
        if c not in seen:
            seen.add(c)
            final_cols.append(c)

    return sql, params, final_cols
