# sql_builder.py
from typing import Tuple, List, Any, Dict


def build_sql(intent: Dict, table: str) -> Tuple[str, List[Any]]:
    """
    Convert generic intent into parameterised SQL + params list.

    intent =
    {
      "select": [
        {"column": "JobProfit", "aggregation": "sum", "alias": "value"}
      ],
      "filters": [
        {"column": "FinancialYear", "operator": "=", "value": 2024},
        {"column": "TransportMode", "operator": "in", "value": ["SEA", "AIR"]}
      ],
      "group_by": ["TransportMode"],
      "order_by": [{"column": "value", "direction": "desc"}],
      "top": 5
    }
    """

    select_items = intent.get("select") or []
    filters = intent.get("filters") or []
    group_by_cols = intent.get("group_by") or []
    order_by_spec = intent.get("order_by") or []
    top_n = intent.get("top")

    params: List[Any] = []

    # ---------------- SELECT ----------------
    select_clauses: List[str] = []

    # Group-by columns first (dimensions)
    for col in group_by_cols:
        select_clauses.append(f"[{col}]")

    # Aggregated metrics
    for s in select_items:
        col = s.get("column")
        agg = (s.get("aggregation") or "").upper()
        alias = s.get("alias") or (f"{agg}_{col}" if agg else col)
        if agg:
            select_clauses.append(f"{agg}([{col}]) AS [{alias}]")
        elif col:
            select_clauses.append(f"[{col}] AS [{alias}]")

    if not select_clauses:
        select_clauses.append("*")

    top_clause = f"TOP {int(top_n)} " if top_n else ""

    sql = f"SELECT {top_clause}" + ", ".join(select_clauses) + f" FROM {table}"

    # ---------------- WHERE ----------------
    if filters:
        where_parts: List[str] = []
        for f in filters:
            col = f.get("column")
            op = (f.get("operator") or "=").lower()
            val = f.get("value")

            if not col:
                continue

            if op == "in" and isinstance(val, (list, tuple)):
                placeholders = ", ".join(["?"] * len(val))
                where_parts.append(f"[{col}] IN ({placeholders})")
                params.extend(val)

            elif op == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
                where_parts.append(f"[{col}] BETWEEN ? AND ?")
                params.extend([val[0], val[1]])

            elif op in ("contains", "like"):
                where_parts.append(f"[{col}] LIKE ?")
                params.append(f"%{val}%")

            elif op == "startswith":
                where_parts.append(f"[{col}] LIKE ?")
                params.append(f"{val}%")

            elif op == "endswith":
                where_parts.append(f"[{col}] LIKE ?")
                params.append(f"%{val}")

            else:
                where_parts.append(f"[{col}] {op.upper()} ?")
                params.append(val)

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

    # ---------------- GROUP BY ----------------
    if group_by_cols:
        group_list = ", ".join(f"[{c}]" for c in group_by_cols)
        sql += f" GROUP BY {group_list}"

    # ---------------- ORDER BY ----------------
    if order_by_spec:
        ob_list: List[str] = []
        for ob in order_by_spec:
            col = ob.get("column")
            if not col:
                continue
            direction = (ob.get("direction") or "DESC").upper()
            # allow both aliases and columns
            ob_list.append(f"[{col}]" if col[0].isalpha() and col[0].isupper() else f"{col} {direction}")
            # but simpler & safe:
            ob_list[-1] = f"{col} {direction}"

        if ob_list:
            sql += " ORDER BY " + ", ".join(ob_list)

    return sql, params
