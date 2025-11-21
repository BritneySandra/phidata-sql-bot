# sql_builder.py
from typing import Dict, Any, List, Tuple


def build_sql(plan: Dict[str, Any], table_name: str) -> Tuple[str, List[Any]]:
    """
    Build a SQL Server query from a generic JSON plan.

    plan format (Option B):
    {
      "select": [
        {"column": "JobProfit", "aggregation": "sum"},
        {"column": "REVAmount", "aggregation": "avg"}
      ],
      "filters": [
        {"column": "FinancialYear", "operator": "=", "value": 2024},
        {"column": "TransportMode", "operator": "in", "value": ["SEA", "AIR"]}
      ],
      "group_by": ["TransportMode"],
      "order_by": [
        {"column": "JobProfit", "direction": "desc"}
      ],
      "limit": 50
    }
    """

    select_items = plan.get("select") or []
    filters = plan.get("filters") or []
    group_by = plan.get("group_by") or []
    order_by = plan.get("order_by") or []
    limit = plan.get("limit")

    params: List[Any] = []
    select_clauses: List[str] = []

    # --- GROUP BY columns must appear in SELECT ---
    if isinstance(group_by, str):
        group_by = [group_by]
    group_by = group_by or []

    for col in group_by:
        select_clauses.append(f"[{col}]")

    # --- SELECT metrics ---
    for item in select_items:
        col = item.get("column")
        agg = (item.get("aggregation") or "").lower()
        if not col:
            continue

        if agg in ("sum", "avg", "max", "min", "count"):
            alias = f"{agg}_{col}"
            select_clauses.append(f"{agg.upper()}([{col}]) AS [{alias}]")
        else:
            # no aggregation, include raw column (only if not already present)
            if col not in group_by:
                select_clauses.append(f"[{col}]")

    if not select_clauses:
        # absolute fallback if LLM gave nothing: count rows
        select_clauses.append("COUNT(*) AS [row_count]")

    select_sql = ", ".join(select_clauses)

    # --- WHERE ---
    where_clauses: List[str] = []
    for f in filters:
        col = f.get("column")
        op = (f.get("operator") or "=").lower()
        val = f.get("value")

        if not col:
            continue

        if op == "in" and isinstance(val, list):
            if not val:
                continue
            placeholders = ",".join("?" for _ in val)
            where_clauses.append(f"[{col}] IN ({placeholders})")
            params.extend(val)

        elif op == "between" and isinstance(val, list) and len(val) == 2:
            where_clauses.append(f"[{col}] BETWEEN ? AND ?")
            params.extend(val)

        elif op == "like":
            where_clauses.append(f"[{col}] LIKE ?")
            params.append(val)

        else:
            if op not in ("=", "!=", "<", ">", "<=", ">="):
                op = "="
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # --- GROUP BY ---
    group_sql = ""
    if group_by:
        gb_cols = ", ".join(f"[{c}]" for c in group_by)
        group_sql = f" GROUP BY {gb_cols}"

    # --- ORDER BY ---
    order_sql = ""
    if order_by:
        if isinstance(order_by, dict):
            order_by = [order_by]
        order_clauses: List[str] = []
        for ob in order_by:
            col = ob.get("column")
            if not col:
                continue
            direction = (ob.get("direction") or "asc").lower()
            dir_sql = "DESC" if direction in ("desc", "descending") else "ASC"
            order_clauses.append(f"[{col}] {dir_sql}")
        if order_clauses:
            order_sql = " ORDER BY " + ", ".join(order_clauses)

    # --- LIMIT (SQL Server uses TOP) ---
    top_prefix = ""
    if isinstance(limit, int) and limit > 0:
        top_prefix = f"TOP {limit} "

    sql = f"SELECT {top_prefix}{select_sql} FROM {table_name} WHERE {where_sql}{group_sql}{order_sql};"

    return sql, params
