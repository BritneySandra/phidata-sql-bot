# sql_builder.py
from typing import Tuple, List, Dict, Any

def build_sql_from_plan(
    plan: Dict[str, Any],
    table: str,
    schema: Dict[str, str]
) -> Tuple[str, List[Any], List[str]]:
    """
    Convert a generic plan from agent.extract_query into:
      - SQL text with parameter placeholders
      - list of parameter values
      - list of output column aliases (for table rendering)
    """

    selects = plan.get("select") or []
    filters = plan.get("filters") or []
    group_by = plan.get("group_by") or []
    order_by = plan.get("order_by") or []
    limit = plan.get("limit")

    if not selects:
        raise ValueError("No valid select expressions in plan")

    select_sql_parts: List[str] = []
    params: List[Any] = []
    output_columns: List[str] = []

    # -------- SELECT --------
    for s in selects:
        if not isinstance(s, dict):
            continue
        col = s.get("column")
        expr = s.get("expression")
        agg = (s.get("aggregation") or "sum").upper()
        alias = s.get("alias") or (col or "value")

        if expr:
            inner = expr  # already proper SQL expression (e.g. [REVAmount] + [WIPAmount])
        elif col:
            inner = f"[{col}]"
        else:
            continue

        if agg in ("SUM", "AVG", "MAX", "MIN", "COUNT"):
            select_piece = f"{agg}({inner}) AS [{alias}]"
        else:
            # no aggregation (rare)
            select_piece = f"{inner} AS [{alias}]"

        select_sql_parts.append(select_piece)
        output_columns.append(alias)

    if not select_sql_parts:
        raise ValueError("No valid select expressions in plan")

    select_clause = ", ".join(select_sql_parts)

    # -------- WHERE --------
    where_clauses: List[str] = []
    for f in filters:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        op = (f.get("operator") or "=").upper()
        val = f.get("value")
        if not col:
            continue

        if op == "IN" and isinstance(val, (list, tuple)):
            placeholders = ",".join("?" for _ in val)
            where_clauses.append(f"[{col}] IN ({placeholders})")
            params.extend(list(val))
        elif op == "BETWEEN" and isinstance(val, (list, tuple)) and len(val) == 2:
            where_clauses.append(f"[{col}] BETWEEN ? AND ?")
            params.extend([val[0], val[1]])
        else:
            where_clauses.append(f"[{col}] {op} ?")
            params.append(val)

    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

    # -------- GROUP BY --------
    if group_by:
        group_cols = [f"[{c}]" for c in group_by if c]
        group_clause = " GROUP BY " + ", ".join(group_cols)
    else:
        group_clause = ""

    # -------- ORDER BY --------
    if order_by:
        ob_parts = []
        for ob in order_by:
            if not isinstance(ob, dict):
                continue
            col = ob.get("column")
            if not col:
                continue
            direction = (ob.get("direction") or "DESC").upper()
            if direction not in ("ASC", "DESC"):
                direction = "DESC"
            ob_parts.append(f"[{col}] {direction}")
        order_clause = " ORDER BY " + ", ".join(ob_parts) if ob_parts else ""
    else:
        order_clause = ""

    # -------- LIMIT / TOP N --------
    # For SQL Server we use OFFSET/FETCH (works in modern versions)
    if isinstance(limit, int) and limit > 0:
        limit_clause = f" OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
    else:
        limit_clause = ""

    sql = (
        f"SELECT {select_clause} "
        f"FROM {table} "
        f"WHERE {where_clause}"
        f"{group_clause}"
        f"{order_clause}"
        f"{limit_clause}"
    )

    return sql, params, output_columns
