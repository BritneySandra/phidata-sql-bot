# sql_builder.py
from typing import Dict, Any, List, Tuple

def _safe_col(col: str) -> str:
    """Wrap column in SQL Server identifier brackets."""
    return f"[{col}]"

def build_sql_from_plan(plan: Dict[str, Any],
                        table: str,
                        schema: Dict[str, str]) -> Tuple[str, List[Any], List[str]]:
    """
    Convert generic plan into SQL Server query + parameters + output columns.
    """
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a dict")

    select_items = plan.get("select") or []
    group_by_cols = plan.get("group_by") or []
    filters = plan.get("filters") or []
    order_by = plan.get("order_by") or []
    limit = plan.get("limit")

    # --- SELECT ---
    output_columns: List[str] = []
    select_sql_parts: List[str] = []

    # 1) group_by columns first (dimension columns)
    for col in group_by_cols:
        if col in schema:
            select_sql_parts.append(_safe_col(col))
            output_columns.append(col)

    # 2) measure columns (aggregations)
    if not select_items:
        raise ValueError("Plan has no select items")

    for idx, sel in enumerate(select_items):
        col = sel.get("column")
        if col not in schema:
            continue
        agg = (sel.get("aggregation") or "").lower()
        alias = sel.get("alias")
        if not alias:
            alias = f"{agg}_{col}" if agg else col
        # If this is the only measure, prefer alias "value" for convenience
        if len(select_items) == 1 and agg and alias not in output_columns:
            alias = "value"
        if agg:
            select_sql_parts.append(f"{agg.upper()}({_safe_col(col)}) AS [{alias}]")
        else:
            select_sql_parts.append(f"{_safe_col(col)} AS [{alias}]")
        output_columns.append(alias)

    if not select_sql_parts:
        raise ValueError("No valid select expressions in plan")

    # TOP N
    top_clause = ""
    if isinstance(limit, int) and limit > 0:
        top_clause = f"TOP {limit} "

    sql = f"SELECT {top_clause}" + ", ".join(select_sql_parts) + f" FROM {table}"

    # --- WHERE ---
    where_clauses: List[str] = []
    params: List[Any] = []

    for f in filters:
        col = f.get("column")
        if col not in schema:
            continue
        op = (f.get("operator") or "=").lower()
        val = f.get("value")

        if op == "in" and isinstance(val, (list, tuple, set)):
            vals = list(val)
            placeholders = ",".join("?" for _ in vals)
            where_clauses.append(f"{_safe_col(col)} IN ({placeholders})")
            params.extend(vals)
        elif op == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
            where_clauses.append(f"{_safe_col(col)} BETWEEN ? AND ?")
            params.extend([val[0], val[1]])
        else:
            if op not in ("=", "!=", "<>", ">", ">=", "<", "<=", "like"):
                op = "="
            where_clauses.append(f"{_safe_col(col)} {op.upper()} ?")
            params.append(val)

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # --- GROUP BY ---
    if group_by_cols:
        gb_cols = [_safe_col(c) for c in group_by_cols if c in schema]
        if gb_cols:
            sql += " GROUP BY " + ", ".join(gb_cols)

    # --- ORDER BY ---
    order_parts: List[str] = []
    for ob in order_by:
        col = ob.get("column")
        direction = (ob.get("direction") or "desc").upper()
        if direction not in ("ASC", "DESC"):
            direction = "DESC"
        # allow ordering by alias or group_by col
        if col in output_columns:
            order_parts.append(f"[{col}] {direction}")
        elif col in group_by_cols:
            order_parts.append(f"{_safe_col(col)} {direction}")

    if order_parts:
        sql += " ORDER BY " + ", ".join(order_parts)

    return sql, params, output_columns
