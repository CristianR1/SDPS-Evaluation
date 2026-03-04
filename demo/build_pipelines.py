"""
build_pipelines.py - Two-Tier Semantic Execution Pipeline Builder

Tier 1: Regex extraction of SQL structure (clauses, predicates, conditionals)
Tier 2: LLM translation of extracted items into natural language instructions

Reads questions.json, parses each SQL query, builds a semantic execution
pipeline format string, and writes it back.

Usage:
    python build_pipelines.py [--no-llm] [--dry-run] [--limit N]
"""
import json
import re
import os
import sys
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

QUESTIONS_PATH = (Path(__file__).resolve().parent / "questions.json").resolve()

# JOIN Logic

def _col_display(col: str) -> str:
    """Human-readable column name: underscores to spaces."""
    return col.replace("_", " ").strip()


def extract_join_pairs_from_sql(sql: str):
    if not sql or not isinstance(sql, str):
        return []
    sql_norm = re.sub(r"\s+", " ", sql).strip()
    alias_to_table = {}
    pairs = []
    from_match = re.search(
        r"\bFROM\s+([a-zA-Z0-9_]+)(?:\s+AS\s+([a-zA-Z0-9_]+))?\s+",
        sql_norm, re.IGNORECASE,
    )
    if not from_match:
        return []
    first_table = from_match.group(1).lower()
    first_alias = (from_match.group(2) or first_table).lower()
    alias_to_table[first_alias] = first_table
    alias_to_table[first_table] = first_table
    rest = sql_norm[from_match.end():]
    join_pat = re.compile(
        r"(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|JOIN)\s+"
        r"([a-zA-Z0-9_]+)(?:\s+AS\s+([a-zA-Z0-9_]+))?\s+ON\s+",
        re.IGNORECASE,
    )
    pos = 0
    while True:
        m = join_pat.search(rest, pos)
        if not m:
            break
        jtype = m.group(1).upper().replace(" ", "")
        if jtype == "LEFTJOIN":
            jtype = "LEFT"
        elif jtype == "RIGHTJOIN":
            jtype = "RIGHT"
        else:
            jtype = "INNER"
        rt = m.group(2).lower()
        ra = (m.group(3) or rt).lower()
        alias_to_table[ra] = rt
        alias_to_table[rt] = rt
        on_start = m.end()
        depth, i = 0, on_start
        while i < len(rest):
            if rest[i] == "(":
                depth += 1
            elif rest[i] == ")":
                depth -= 1
            elif depth == 0 and i + 5 <= len(rest):
                sub = rest[i:i + 6].upper()
                if any(sub.startswith(kw) for kw in
                       (" INNER", " LEFT", " RIGHT", " JOIN",
                        " WHERE", " GROUP", " ORDER", " HAVIN")):
                    break
            i += 1
        on_clause = rest[on_start:i]
        refs = re.findall(r"\b([a-zA-Z0-9_]+)\s*\.", on_clause)
        left_ref = None
        for r in refs:
            rl = r.lower()
            if rl in alias_to_table and alias_to_table[rl] != rt:
                left_ref = rl
                break
        lt = alias_to_table.get(left_ref, list(alias_to_table.values())[0]) if left_ref else list(alias_to_table.values())[0]
        col_m = re.search(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", on_clause)
        lc = rc = None
        if col_m:
            a1, c1, a2, c2 = col_m.groups()
            t1 = alias_to_table.get(a1.lower())
            t2 = alias_to_table.get(a2.lower())
            if t1 and t2:
                if t1 == lt and t2 == rt:
                    lc, rc = c1, c2
                elif t1 == rt and t2 == lt:
                    lc, rc = c2, c1
                else:
                    lc, rc = c1, c2
        pairs.append((jtype, lt, rt, lc, rc))
        pos = m.end()
    return pairs


def build_join_ops(sql: str) -> list[str]:
    pairs = extract_join_pairs_from_sql(sql)
    ops = []
    for jtype, _lt, _rt, lc, rc in pairs:
        if lc and rc:
            instr = (f"The {_col_display(lc)} described in one document is the "
                     f"same as the {_col_display(rc)} described in the other document.")
        else:
            instr = ("The key described in one document is the same as "
                     "the key described in the other document.")
        prefix = {"LEFT": "LEFT JOIN", "RIGHT": "RIGHT JOIN"}.get(jtype, "JOIN")
        ops.append(f"{prefix} - {instr}")
    return ops


# SQL Utilities (alias, subquery detection and decomposition functions)

def build_alias_map(sql: str) -> dict:
    """Alias -> real table name (lowercase)."""
    sql_norm = re.sub(r"\s+", " ", sql).strip()
    amap = {}
    fm = re.search(r"\bFROM\s+(\w+)(?:\s+AS\s+(\w+))?", sql_norm, re.IGNORECASE)
    if fm:
        t, a = fm.group(1).lower(), (fm.group(2) or fm.group(1)).lower()
        amap[a] = t
        amap[t] = t
    for m in re.finditer(
            r"(?:INNER\s+|LEFT\s+|RIGHT\s+)?JOIN\s+(\w+)(?:\s+AS\s+(\w+))?",
            sql_norm, re.IGNORECASE):
        t, a = m.group(1).lower(), (m.group(2) or m.group(1)).lower()
        amap[a] = t
        amap[t] = t
    return amap


def has_subquery(sql: str) -> bool:
    """Detect subqueries, CTEs, UNION, EXCEPT that make pipeline translation infeasible."""
    up = sql.upper().strip()
    if up.startswith("WITH "):
        return True
    if re.search(r"\bUNION\b|\bEXCEPT\b|\bINTERSECT\b", up):
        return True
    depth = 0
    for i, c in enumerate(sql):
        if c == "(":
            depth += 1
            rest = sql[i + 1:].lstrip()
            if rest.upper().startswith("SELECT"):
                return True
        elif c == ")":
            depth -= 1
    return False


def _kw_at_depth0(sql_norm: str, keyword: str, start: int = 0) -> int:
    """Return position of *keyword* at parenthesis depth 0, or -1."""
    kw_len = len(keyword)
    depth = 0
    i = start
    while i < len(sql_norm):
        c = sql_norm[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0:
            if sql_norm[i:i + kw_len].upper() == keyword.upper():
                before_ok = (i == 0 or not sql_norm[i - 1].isalnum() and sql_norm[i - 1] != "_")
                after_ok = (i + kw_len >= len(sql_norm) or
                            not sql_norm[i + kw_len].isalnum() and sql_norm[i + kw_len] != "_")
                if before_ok and after_ok:
                    return i
        i += 1
    return -1


def _split_select_on_commas(select_str: str) -> list[str]:
    """Split SELECT body on top-level commas to get individual expressions.
    Regex: comma at parenthesis depth 0 (handles ' , ' or ',').
    Number of commas + 1 = number of expected answers / pipelines required."""
    if not select_str or not select_str.strip():
        return [select_str] if select_str else []
    # Use _split_depth0 to split on comma only at depth 0
    parts = _split_depth0(select_str.strip(), ",")
    return [p.strip() for p in parts if p.strip()]


def extract_sql_clauses(sql: str) -> dict:
    """Extract top-level clause bodies: SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT."""
    sql_norm = re.sub(r"\s+", " ", sql).strip()
    kw_list = [
        ("SELECT", "SELECT"),
        ("FROM", "FROM"),
        ("WHERE", "WHERE"),
        ("GROUP BY", "GROUP BY"),
        ("HAVING", "HAVING"),
        ("ORDER BY", "ORDER BY"),
        ("LIMIT", "LIMIT"),
    ]
    positions = {}
    for name, kw in kw_list:
        pos = _kw_at_depth0(sql_norm, kw)
        if pos >= 0:
            positions[name] = (pos, pos + len(kw))

    sorted_kws = sorted(positions.items(), key=lambda x: x[1][0])
    clauses = {}
    for idx, (name, (_, content_start)) in enumerate(sorted_kws):
        end = sorted_kws[idx + 1][1][0] if idx + 1 < len(sorted_kws) else len(sql_norm)
        clauses[name] = sql_norm[content_start:end].strip()
    return clauses


# Low-Level Parsing Helpers

def _split_depth0(s: str, sep: str) -> list[str]:
    """Split *s* on *sep* only at parenthesis depth 0."""
    parts, buf, depth = [], [], 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
            buf.append(c)
        elif c == ")":
            depth -= 1
            buf.append(c)
        elif depth == 0 and s[i:i + len(sep)] == sep:
            parts.append("".join(buf).strip())
            buf = []
            i += len(sep)
            continue
        else:
            buf.append(c)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _split_top_and(clause: str) -> list[str]:
    """Split on AND at depth 0, respecting BETWEEN...AND."""
    parts, buf, depth, between_pending = [], [], 0, False
    i = 0
    while i < len(clause):
        c = clause[i]
        if c == "(":
            depth += 1
            buf.append(c)
            i += 1
            continue
        if c == ")":
            depth -= 1
            buf.append(c)
            i += 1
            continue
        if depth == 0:
            rest = clause[i:].upper()
            if rest.startswith("BETWEEN") and (i == 0 or not clause[i - 1].isalnum()):
                after = i + 7
                if after >= len(clause) or not clause[after].isalnum():
                    between_pending = True
            if rest.startswith("AND") and (i == 0 or not clause[i - 1].isalnum()):
                after = i + 3
                if after >= len(clause) or not clause[after].isalnum():
                    if between_pending:
                        between_pending = False
                        buf.append(clause[i:i + 3])
                        i += 3
                        continue
                    else:
                        parts.append("".join(buf).strip())
                        buf = []
                        i += 3
                        continue
        buf.append(c)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _resolve_col(expr: str, alias_map: dict) -> tuple[str | None, str]:
    """Resolve alias.column or function(alias.column) to (table, column).
    Unwraps SUBSTR(), STRFTIME(), CAST() etc. to find the inner column."""
    # Backtick-quoted column with alias: T2.`T-BIL`, T2.`Examination Date`
    m = re.search(r"(\w+)\.`([^`]+)`", expr)
    if m:
        alias = m.group(1).lower()
        col = m.group(2)
        return alias_map.get(alias, alias), col
    m = re.search(r"(\w+)\.(\w+)", expr)
    if m:
        alias = m.group(1).lower()
        col = m.group(2)
        return alias_map.get(alias, alias), col
    # Bare column name
    m = re.match(r"`?(\w+)`?$", expr.strip())
    if m:
        return None, m.group(1)
    # Function wrapping a bare column: SUBSTR(Date, ...), STRFTIME('%Y', Birthday)
    m = re.search(r"\b\w+\s*\(\s*`?(\w+)`?[\s,)]", expr)
    if m:
        return None, m.group(1)
    return None, expr.strip()


_MONTH_NAMES = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}


def _humanize_date(val: str) -> str:
    """Convert compact date values to natural language with month names.

    201309     → September 2013
    2012-08-24 → August 24, 2012
    1981-11-   → November 1981
    1996-01    → January 1996
    Non-date values pass through unchanged.
    """
    val = val.strip()
    # YYYYMM  (compact 6-digit)
    m = re.match(r"^(\d{4})(0[1-9]|1[0-2])$", val)
    if m:
        month = _MONTH_NAMES.get(m.group(2))
        if month:
            return f"{month} {m.group(1)}"
    # YYYY-MM-DD
    m = re.match(r"^(\d{4})-(0[1-9]|1[0-2])-(\d{2})$", val)
    if m:
        month = _MONTH_NAMES.get(m.group(2))
        if month:
            day = str(int(m.group(3)))
            return f"{month} {day}, {m.group(1)}"
    # YYYY-MM- (trailing dash from LIKE prefix)
    m = re.match(r"^(\d{4})-(0[1-9]|1[0-2])-$", val)
    if m:
        month = _MONTH_NAMES.get(m.group(2))
        if month:
            return f"{month} {m.group(1)}"
    # YYYY-MM (no day)
    m = re.match(r"^(\d{4})-(0[1-9]|1[0-2])$", val)
    if m:
        month = _MONTH_NAMES.get(m.group(2))
        if month:
            return f"{month} {m.group(1)}"
    return val


# Predicate Parsing (WHERE / HAVING)

def _like_template(col: str, pattern: str, negate: bool = False) -> str:
    cd = _col_display(col)
    if pattern.startswith("%") and pattern.endswith("%"):
        verb = "does not mention" if negate else "mentions"
        return f"The {cd} {verb} {_humanize_date(pattern.strip('%'))}"
    if pattern.endswith("%"):
        stripped = pattern.rstrip("%")
        humanized = _humanize_date(stripped)
        if humanized != stripped:
            verb = "is not in" if negate else "is in"
            return f"The {cd} {verb} {humanized}"
        return f"The {cd} starts with {stripped}"
    if pattern.startswith("%"):
        return f"The {cd} ends with {_humanize_date(pattern.lstrip('%'))}"
    verb = "is not like" if negate else "is like"
    return f"The {cd} {verb} {_humanize_date(pattern)}"


def classify_predicate(pred: str, alias_map: dict) -> dict:
    """Classify a single WHERE/HAVING predicate and produce a Tier-1 template."""
    pred = pred.strip()
    res = {"raw": pred, "op": None, "col": None, "table": None,
           "val": None, "val2": None, "template": pred}

    # Compound conditions (AND/OR at depth 0) — pass through as-is for LLM
    if re.search(r"\bAND\b|\bOR\b", pred, re.I):
        inner_and = _split_top_and(pred)
        if len(inner_and) > 1:
            sub_templates = []
            for sub in inner_and:
                sp = classify_predicate(sub, alias_map)
                sub_templates.append(sp["template"])
            res["template"] = " and ".join(sub_templates)
            return res

    # IS NOT NULL
    m = re.match(r"(.+?)\s+IS\s+NOT\s+NULL\s*$", pred, re.I)
    if m:
        t, c = _resolve_col(m.group(1), alias_map)
        res.update(op="IS NOT NULL", col=c, table=t,
                   template=f"The {_col_display(c)} has a value")
        return res

    # IS NULL
    m = re.match(r"(.+?)\s+IS\s+NULL\s*$", pred, re.I)
    if m:
        t, c = _resolve_col(m.group(1), alias_map)
        res.update(op="IS NULL", col=c, table=t,
                   template=f"The {_col_display(c)} is absent")
        return res

    # NOT BETWEEN
    m = re.match(r"(.+?)\s+NOT\s+BETWEEN\s+(.+?)\s+AND\s+(.+?)\s*$", pred, re.I)
    if m:
        t, c = _resolve_col(m.group(1), alias_map)
        v1 = m.group(2).strip().strip("'\"")
        v2 = m.group(3).strip().strip("'\"")
        res.update(op="NOT BETWEEN", col=c, table=t, val=v1, val2=v2,
                   template=f"The {_col_display(c)} is not between {_humanize_date(v1)} and {_humanize_date(v2)}")
        return res

    # BETWEEN
    m = re.match(r"(.+?)\s+BETWEEN\s+(.+?)\s+AND\s+(.+?)\s*$", pred, re.I)
    if m:
        t, c = _resolve_col(m.group(1), alias_map)
        v1 = m.group(2).strip().strip("'\"")
        v2 = m.group(3).strip().strip("'\"")
        res.update(op="BETWEEN", col=c, table=t, val=v1, val2=v2,
                   template=f"The {_col_display(c)} is between {_humanize_date(v1)} and {_humanize_date(v2)}")
        return res

    # NOT LIKE
    m = re.match(r"(.+?)\s+NOT\s+LIKE\s+(.+?)\s*$", pred, re.I)
    if m:
        t, c = _resolve_col(m.group(1), alias_map)
        pat = m.group(2).strip().strip("'\"")
        res.update(op="NOT LIKE", col=c, table=t, val=pat,
                   template=_like_template(c, pat, negate=True))
        return res

    # LIKE
    m = re.match(r"(.+?)\s+LIKE\s+(.+?)\s*$", pred, re.I)
    if m:
        t, c = _resolve_col(m.group(1), alias_map)
        pat = m.group(2).strip().strip("'\"")
        res.update(op="LIKE", col=c, table=t, val=pat,
                   template=_like_template(c, pat))
        return res

    # NOT IN
    m = re.match(r"(.+?)\s+NOT\s+IN\s*\((.+?)\)\s*$", pred, re.I)
    if m:
        t, c = _resolve_col(m.group(1), alias_map)
        res.update(op="NOT IN", col=c, table=t, val=m.group(2).strip(),
                   template=f"The {_col_display(c)} is not in ({m.group(2).strip()})")
        return res

    # IN
    m = re.match(r"(.+?)\s+IN\s*\((.+?)\)\s*$", pred, re.I)
    if m:
        t, c = _resolve_col(m.group(1), alias_map)
        res.update(op="IN", col=c, table=t, val=m.group(2).strip(),
                   template=f"The {_col_display(c)} is in ({m.group(2).strip()})")
        return res

    # Comparison operators (multi-char first)
    for regex_op, name, text in [
        (r">=", ">=", "is greater than or equal to"),
        (r"<=", "<=", "is less than or equal to"),
        (r"<>", "<>", "is not"),
        (r"!=", "!=", "is not"),
        (r">",  ">",  "is greater than"),
        (r"<",  "<",  "is less than"),
        (r"=",  "=",  "is"),
    ]:
        m = re.match(rf"(.+?)\s*{regex_op}\s*(.+?)\s*$", pred)
        if m:
            lhs = m.group(1).strip()
            rhs = m.group(2).strip().strip("'\"")
            t, c = _resolve_col(lhs, alias_map)
            res.update(op=name, col=c, table=t, val=rhs,
                       template=f"The {_col_display(c)} {text} {_humanize_date(rhs)}")
            return res

    return res


def parse_where(where_str: str, alias_map: dict) -> list[dict]:
    if not where_str:
        return []
    preds = _split_top_and(where_str)
    return [classify_predicate(p, alias_map) for p in preds if p.strip()]


# Section 5: SELECT Parsing

def _extract_iif_inner(select_str: str) -> list[dict]:
    """Extract all IIF(cond, true_val, false_val) occurrences from SELECT."""
    results = []
    for m in re.finditer(r"\bIIF\s*\(", select_str, re.I):
        start = m.end()
        depth, i = 1, start
        while i < len(select_str) and depth > 0:
            if select_str[i] == "(":
                depth += 1
            elif select_str[i] == ")":
                depth -= 1
            i += 1
        inner = select_str[start:i - 1]
        parts = _split_depth0(inner, ",")
        if len(parts) >= 2:
            results.append({
                "condition": parts[0].strip(),
                "true_value": parts[1].strip(),
                "false_value": parts[2].strip() if len(parts) > 2 else "0",
            })
    return results


def _extract_case_when(select_str: str) -> list[dict]:
    """Extract CASE WHEN cond THEN val constructs."""
    results = []
    for m in re.finditer(r"\bCASE\s+WHEN\s+(.+?)\s+THEN\s+(.+?)(?:\s+ELSE|\s+END)",
                         select_str, re.I):
        results.append({
            "condition": m.group(1).strip(),
            "true_value": m.group(2).strip(),
            "false_value": "0",
        })
    return results


def _extract_boolean_sums(select_str: str) -> list[dict]:
    """Detect SQLite SUM(comparison) patterns — equivalent to SUM(IIF(cmp,1,0))."""
    results = []
    for m in re.finditer(r"\bSUM\s*\(", select_str, re.I):
        start = m.end()
        depth, i = 1, start
        while i < len(select_str) and depth > 0:
            if select_str[i] == "(":
                depth += 1
            elif select_str[i] == ")":
                depth -= 1
            i += 1
        inner = select_str[start:i - 1].strip()
        if re.search(r"\bIIF\b|\bCASE\b", inner, re.I):
            continue
        if re.search(r"[=<>!]|\bLIKE\b|\bBETWEEN\b|\bIS\b", inner, re.I):
            results.append({
                "condition": inner,
                "true_value": "1",
                "false_value": "0",
            })
    return results


# Percent count denominator spec (for lotus_pipeline / palimpzest_pipeline):
# The denominator = count at the split point:
#   - If there are JOIN operators: count after the last join operation.
#   - If no joins: count of the original document set (starting amount).
# Note: Palimpzest semantic joins produce a larger document set than SQL
# (e.g. join expansion). If denominators inflate use original doc set count.

def _detect_percent_denominator_type(select_str: str, bare_aggs: list[dict]) -> str | None:
    """Detect denominator type for * 100 / pattern: 'count' or 'sum'.
    percent count: denominator is COUNT(...) — majority case.
    percent sum: denominator is SUM(column) — percent of sum over total sum."""
    if not re.search(r"\*\s*100\s*/", select_str):
        return None
    # Find the token after " * 100 / " — should be COUNT or SUM
    m = re.search(r"\*\s*100\s*/\s*(?:CAST\s*\()?(SUM|COUNT)\s*\(", select_str, re.I)
    if m:
        return "count" if m.group(1).upper() == "COUNT" else "sum"
    # Fallback: check bare_aggs — for single IIF + bare agg, denominator is the bare agg
    sum_aggs = [a for a in bare_aggs if a["func"].upper() == "SUM"]
    count_aggs = [a for a in bare_aggs if a["func"].upper() == "COUNT"]
    if count_aggs and not sum_aggs:
        return "count"
    if sum_aggs and not count_aggs:
        return "sum"
    return "count"  # default to count (majority case)


def _detect_outer_arithmetic(select_str: str) -> str | None:
    """Detect the arithmetic operator between aggregate terms in SELECT."""
    if re.search(r"\*\s*100\s*/", select_str):
        return "percent"
    depth = 0
    for i, c in enumerate(select_str):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and c == "-":
            before = select_str[:i].rstrip()
            after = select_str[i + 1:].lstrip()
            if (re.search(r"\)$", before) and
                    re.search(r"^(?:CAST\s*\()?(?:SUM|COUNT|AVG|MAX|MIN)\b", after, re.I)):
                return "-"
        elif depth == 0 and c == "/":
            before = select_str[:i].rstrip()
            after = select_str[i + 1:].lstrip()
            if (re.search(r"\)$", before) and
                    re.search(r"^(?:CAST\s*\()?(?:SUM|COUNT|AVG|MAX|MIN)\b", after, re.I)):
                return "/"
    return None


def _extract_output_columns(select_str: str, alias_map: dict) -> list[dict]:
    """Identify bare (non-aggregate, non-IIF) columns in SELECT for EXTRACT."""
    cols = _split_depth0(select_str, ",")
    out = []
    for raw_col in cols:
        c = raw_col.strip()
        if re.search(r"\b(SUM|AVG|COUNT|MAX|MIN|IIF|CASE)\s*[\(]", c, re.I):
            continue
        if re.search(r"\*\s*100\s*/", c):
            continue
        c_clean = re.sub(r"^\s*DISTINCT\s+", "", c, flags=re.I)
        c_clean = re.sub(r"\s+AS\s+\w+\s*$", "", c_clean, flags=re.I).strip()
        t, col = _resolve_col(c_clean, alias_map)
        out.append({"table": t, "column": col, "raw": raw_col.strip()})
    return out


def _detect_bare_aggregates(select_str: str) -> list[dict]:
    """Find aggregate calls NOT wrapping IIF/CASE (bare aggregates)."""
    out = []
    for func in ("SUM", "AVG", "COUNT", "MAX", "MIN"):
        for m in re.finditer(rf"\b{func}\s*\(", select_str, re.I):
            start = m.end()
            depth, i = 1, start
            while i < len(select_str) and depth > 0:
                if select_str[i] == "(":
                    depth += 1
                elif select_str[i] == ")":
                    depth -= 1
                i += 1
            inner = select_str[start:i - 1].strip()
            if re.search(r"\bIIF\b|\bCASE\b", inner, re.I):
                continue
            if re.search(r"[=<>!]|\bLIKE\b|\bBETWEEN\b", inner, re.I):
                continue
            out.append({"func": func, "column": inner})
    return out


def parse_select(select_str: str, alias_map: dict) -> dict:
    """Full SELECT analysis: IIF branches, arithmetic, aggregates, output cols."""
    iif_all = (_extract_iif_inner(select_str) +
               _extract_case_when(select_str) +
               _extract_boolean_sums(select_str))

    # Deduplicate IIF branches by condition text (normalized)
    seen_conds = set()
    deduped = []
    for branch in iif_all:
        norm = re.sub(r"\s+", " ", branch["condition"]).strip().upper()
        if norm in seen_conds:
            continue
        seen_conds.add(norm)
        tv = branch["true_value"]
        branch["returns_column"] = tv not in ("1", "0", "'1'", "'0'",
                                               "true", "false", "'YES'", "'NO'")
        deduped.append(branch)
    iif_all = deduped

    arithmetic = _detect_outer_arithmetic(select_str)
    bare_aggs = _detect_bare_aggregates(select_str)
    output_cols = _extract_output_columns(select_str, alias_map)

    has_branching = (len(iif_all) >= 2 or
                     (len(iif_all) == 1 and (bare_aggs or arithmetic)))

    percent_denom = _detect_percent_denominator_type(select_str, bare_aggs) if arithmetic == "percent" else None

    return {
        "iif_branches": iif_all,
        "arithmetic": arithmetic,
        "bare_aggregates": bare_aggs,
        "output_columns": output_cols,
        "has_branching": has_branching,
        "percent_denominator_type": percent_denom,
    }


# GROUP BY / ORDER BY / LIMIT

def parse_group_by(group_str: str, alias_map: dict) -> list[dict]:
    if not group_str:
        return []
    cols = _split_depth0(group_str, ",")
    out = []
    seen = set()
    for raw in cols:
        t, c = _resolve_col(raw.strip(), alias_map)
        key = (t, c)
        if key in seen:
            continue
        seen.add(key)
        out.append({"table": t, "column": c, "raw": raw.strip(),
                    "template": f"Group by {_col_display(c)}"})
    return out


def parse_order_by_limit(order_str: str, limit_str: str, alias_map: dict) -> dict | None:
    if not order_str:
        return None
    first = _split_depth0(order_str, ",")[0].strip()
    direction = "ASC"
    if re.search(r"\bDESC\b", first, re.I):
        direction = "DESC"
    first_clean = re.sub(r"\s+(ASC|DESC)\b", "", first, flags=re.I).strip()

    # Unwrap aggregate: SUM(T2.Consumption) → resolve inner column
    agg_m = re.match(r"(SUM|AVG|COUNT|MAX|MIN)\s*\((.+)\)$", first_clean, re.I)
    if agg_m:
        inner = agg_m.group(2).strip()
        t, c = _resolve_col(inner, alias_map)
    else:
        t, c = _resolve_col(first_clean, alias_map)

    k = None
    if limit_str:
        lm = re.match(r"(\d+)", limit_str.strip())
        if lm:
            k = int(lm.group(1))

    cd = _col_display(c)
    if k == 1:
        word = "highest" if direction == "DESC" else "lowest"
        template = f"The one with {word} {cd}"
    elif k and k > 1:
        template = f"Top {k} by {cd} {'descending' if direction == 'DESC' else 'ascending'}"
    else:
        template = f"Order by {cd} {'descending' if direction == 'DESC' else 'ascending'}"

    return {"column": c, "table": t, "direction": direction, "k": k,
            "raw": order_str.strip(), "template": template}


# LLM Translation and Refinement

_LLM_CLIENT = None


def _get_llm_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        try:
            from openai import OpenAI
            _LLM_CLIENT = OpenAI()
        except Exception as e:
            print(f"[WARN] Could not initialize OpenAI client: {e}")
            return None
    return _LLM_CLIENT


_STYLE_EXAMPLES = {
    "FILTER": [
        "The phone is 809-555-3360",
        "The Consumption is greater than 46.73",
        "The Date is between August 2013 and November 2013",
    ],
    "GROUP": [
        "Group by category",
        "Group by CustomerID",
    ],
    "AGGREGATE": [
        "The total gas consumption",
        "The average cost spent",
        "The sum of the amount",
    ],
    "EXTRACT": [
        "The product description",
        "The fundraising notes",
        "The student's major name",
        "The state of the school",
    ],
    "RANK": [
        "The one with lowest Consumption",
        "Order by birthday ascending",
    ],
}


def llm_semantic_instruction(nlq: str, evidence: str, column: str,
                              op_type: str, sql_context: str, client=None) -> str:
    """Generate a semantically rich instruction for EXTRACT/AGGREGATE using NLQ context.

    Unlike llm_translate which cleans raw SQL templates, this function creates
    contextually meaningful instructions by understanding what the NLQ is asking for.
    """
    cl = client or _get_llm_client()
    if cl is None:
        if op_type == "EXTRACT":
            return f"The {_col_display(column)}"
        return f"{_col_display(column)}"

    examples = _STYLE_EXAMPLES.get(op_type, _STYLE_EXAMPLES["EXTRACT"])
    examples_block = "\n".join(f'  - "{ex}"' for ex in examples)

    if op_type == "EXTRACT":
        task_desc = (
            "Add brief semantic context to this column name for an EXTRACT instruction. "
            "The output must refer ONLY to the specific column provided, not other columns."
            "Do not include non semantic operative text, like difference or sum"
        )
        rules = (
            f"- Output a short phrase (3-8 words) for ONLY the column: {column}\n"
            f"- Add minimal context from the question (e.g., 'State' -> 'The state of the school').\n"
            f"- Do NOT mention other columns or combine multiple fields.\n"
            f"- Do NOT describe filtering, ranking, or other operations.\n"
            f"- NEVER use the words \"document\" or \"set\".\n"
            f"- Output ONLY the instruction text, nothing else."
        )
    else:
        task_desc = (
            "Generate a concise AGGREGATE instruction describing the computation to perform."
            "Do not include non semantic operative text, like percent or difference only totals, sums, and averages"

        )
        rules = (
            f"- Output a short phrase (3-8 words) describing the aggregation.\n"
            f"- Focus on what is being summed/averaged/counted.\n"
            f"- NEVER use the words \"document\" or \"set\".\n"
            f"- Output ONLY the instruction text, nothing else."
        )

    prompt = (
        f"{task_desc}\n\n"
        f"Rules:\n{rules}\n\n"
        f"Style examples for {op_type}:\n{examples_block}\n\n"
        f"Context:\n"
        f"- Question: \"{nlq}\"\n"
        f"- Column: {column}\n\n"
        f"Instruction:"
    )
    try:
        resp = cl.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        result = resp.choices[0].message.content.strip().strip('"').strip("'")
        print(f"  [LLM {op_type}] {column!r} -> {result!r}")
        return result
    except Exception as e:
        print(f"[WARN] LLM semantic instruction failed: {e}")
        if op_type == "EXTRACT":
            return f"The {_col_display(column)}"
        return f"{_col_display(column)}"


def llm_translate(nlq: str, evidence: str, sql_clause: str,
                  op_type: str, template: str, client=None) -> str:
    """Call LLM to refine a Tier-1 template into a natural instruction."""
    cl = client or _get_llm_client()
    if cl is None:
        return template

    examples = _STYLE_EXAMPLES.get(op_type, _STYLE_EXAMPLES["FILTER"])
    examples_block = "\n".join(f'  - "{ex}"' for ex in examples)

    prompt = (
        f"Rewrite a raw SQL-derived template into a clean natural-language instruction.\n\n"
        f"Rules:\n"
        f"- NEVER use the words \"document\" or \"set\" in the output.\n"
        f"- Remove all SQL artifacts: function calls (STRFTIME, SUBSTR, CAST),\n"
        f"  table aliases (T1., T2.), backtick identifiers, leaked quotes, and\n"
        f"  unbalanced parentheses.\n"
        f"- Keep the instruction simple and unitary — only the {op_type} operation,\n"
        f"  no other operations.\n"
        f"- Use the NLQ and evidence to infer human-readable column and value names.\n"
        f"- Output ONLY the instruction text, nothing else.\n\n"
        f"Match this style for {op_type} instructions:\n{examples_block}\n\n"
        f"Context:\n"
        f"- NLQ: \"{nlq}\"\n"
        f"- Evidence: \"{evidence}\"\n"
        f"- SQL clause: {sql_clause}\n"
        f"- Template: \"{template}\"\n\n"
        f"Instruction:"
    )
    try:
        resp = cl.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        result = resp.choices[0].message.content.strip().strip('"').strip("'")
        print(f"  [LLM {op_type}] {template!r} -> {result!r}")
        return result
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}")
        return template


# Section 8: OP Detection & Mapping

KNOWN_OPS = {
    "total",                # COUNT — number of documents in result set
    "ratio",                # Division of two branch counts
    "-",                    # Difference between two branches
    ">",                    # Comparison: branch A > branch B
    "bool",                 # Existence check (True/False)
    "bool: True",           # Existence check with expected value
    "percent count",        # subset_count / total_count * 100 (denominator COUNT)
    "percent sum",          # subset_sum / total_sum * 100 (denominator SUM)
    "percent",              # Generic: numerator / denominator * 100 (was percent forward)
    "total percent",        # filtered_count / total_count * 100
    ">,-",                  # Compare then difference (composite)
    "-, total",             # Difference then count (composite)
    "/, *100",              # Divide then multiply by 100
    "> percent",            # Compare then percentage (composite)
    "percent[p1/p2]",       # Parameterized percentage (pipeline1 / pipeline2)
    "percent",              # Generic percentage (generated by builder)
}


def collect_existing_ops(questions: list) -> set:
    """Scan existing format strings to catalog all OP: types.

    OP values are short tokens (e.g. 'total', 'ratio', '-', 'percent reverse',
    '>,-', 'percent[p1/p2]').  We extract them by finding 'OP:' then capturing
    up to the next structural delimiter:
      &&  |  / (branch open)  " (json boundary)
      , followed by a keyword (FILTER, EXTRACT, JOIN, GROUP, RANK, AGGREGATE, OP)
      or end of string.
    """
    kw_boundary = (r"(?=\s*&&|\s*\||\s*/\s|\s*\""
                   r"|,\s*(?:FILTER|EXTRACT|JOIN|GROUP|RANK|AGGREGATE|OP)\b"
                   r"|\s*$)")
    ops = set()
    for q in questions:
        fmt = q.get("format", "")
        for m in re.finditer(rf"OP:\s*(.+?){kw_boundary}", fmt):
            raw = m.group(1).strip().rstrip('"').strip()
            if raw:
                ops.add(raw)
    return ops


def _nlq_asks_percentage(nlq: str) -> bool:
    return bool(re.search(r"\bpercent(?:age)?\b", nlq, re.I))


def _nlq_asks_ratio(nlq: str) -> bool:
    return bool(re.search(r"\bratio\b", nlq, re.I))


def _nlq_asks_boolean(nlq: str) -> bool:
    """Detect yes/no questions: 'Is it true...', 'Did X attend...', 'Was X...'."""
    low = nlq.strip().lower()
    if re.match(r"^(is it true|did |was |were |are |is |does |do )", low):
        if not re.search(r"\bhow many\b|\bwhat\b|\bhow much\b", low):
            return True
    return False


def _nlq_asks_compare_then_diff(nlq: str) -> bool:
    """Detect compound: 'Is it true that X? If so, how many more?'."""
    low = nlq.lower()
    return bool(
        re.search(r"is it true.*how many more", low) or
        re.search(r"more.*if so.*how many", low)
    )


def _nlq_asks_compare_then_percent(nlq: str) -> bool:
    low = nlq.lower()
    return bool(
        re.search(r"more.*what is the deviation in percentage", low) or
        re.search(r"more.*percent", low) and re.search(r"^(is it true|are there more)", low)
    )


def _select_returns_boolean(select_str: str) -> bool:
    """Check if SELECT's CASE/IIF returns boolean-like literals."""
    return bool(re.search(
        r"THEN\s+['\"]?(YES|NO|true|false|TRUE|FALSE)['\"]?\s",
        select_str, re.I
    )) or bool(re.search(
        r"IIF\s*\([^,]+,\s*['\"]?(YES|NO|true|false)['\"]?\s*,",
        select_str, re.I
    ))


def _llm_determine_op(nlq: str, evidence: str, sql: str,
                       base_op: str, client=None) -> str:
    """Tier 2: Ask LLM to pick the precise OP variant."""
    cl = client or _get_llm_client()
    if cl is None:
        return base_op
    op_choices = sorted(KNOWN_OPS)
    prompt = (
        f"A semantic data pipeline needs a post-processing OP type.\n\n"
        f"NLQ: \"{nlq}\"\n"
        f"Evidence: \"{evidence}\"\n"
        f"SQL (first 400 chars): {sql}\n"
        f"Base OP detected from SQL structure: {base_op}\n\n"
        f"Available OP types and their meanings:\n"
        f"  total          — count documents in the result set\n"
        f"  ratio          — divide count of branch A by count of branch B\n"
        f"  -              — subtract count of branch B from branch A\n"
        f"  >              — compare: is branch A count > branch B count?\n"
        f"  bool           — existence check (does the result set have documents?)\n"
        f"  bool: True     — existence check with expected True\n"
        f"  percent count   — (subset count / total count) * 100, denominator COUNT\n"
        f"  percent sum     — (subset sum / total sum) * 100, denominator SUM\n"
        f"  percent         — (numerator / denominator) * 100, generic\n"
        f"  total percent  — (filtered count / all count) * 100, both via filters\n"
        f"  >,-            — compare then difference (composite)\n"
        f"  -, total       — difference then count (composite)\n"
        f"  /, *100        — divide extracted values then * 100\n"
        f"  > percent      — compare then percentage (composite)\n"
        f"  percent[p1/p2] — percentage of extracted value p1 over p2\n"
        f"  percent        — generic percentage\n\n"
        f"Pick the single most accurate OP type for this query.\n"
        f"Output ONLY the OP type name, nothing else."
    )
    try:
        resp = cl.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.0,
        )
        choice = resp.choices[0].message.content.strip().strip('"').strip("'")
        if choice in KNOWN_OPS:
            return f"OP: {choice}"
        return base_op
    except Exception as e:
        print(f"[WARN] LLM OP refinement failed: {e}")
        return base_op


def determine_op_type(select_info: dict, select_str: str = "",
                      nlq: str = "", evidence: str = "", sql: str = "",
                      use_llm: bool = False, client=None) -> str | None:
    """
    Three-tier OP type determination.

    Tier 1  — SQL structure: arithmetic operators, aggregate types, boolean returns.
    Tier 1.5— NLQ heuristics: keyword patterns distinguish variants.
    Tier 2  — LLM refinement: for ambiguous cases the LLM picks the precise OP.
    """
    arith = select_info["arithmetic"]
    iif_count = len(select_info["iif_branches"])
    bare_aggs = select_info["bare_aggregates"]
    bare_count = len(bare_aggs)
    bare_funcs = {a["func"].upper() for a in bare_aggs}
    bare_has_count = "COUNT" in bare_funcs
    output_cols = select_info["output_columns"]

    if iif_count == 0 and bare_count == 0 and not _select_returns_boolean(select_str):
        if _nlq_asks_boolean(nlq):
            return "OP: bool"
        return None

    if _select_returns_boolean(select_str):
        return "OP: bool"

    if arith == "percent":
        percent_denom = select_info.get("percent_denominator_type")
        if iif_count >= 2:
            # Two IIF sets being compared as percentage — rare, use LLM
            if use_llm:
                return _llm_determine_op(nlq, evidence, sql, "OP: percent", client)
            return "OP: percent"
        if iif_count == 1 and percent_denom == "sum":
            # IIF subset sum vs SUM total — percent sum
            return "OP: percent sum"
        if iif_count == 1 and (bare_has_count or percent_denom == "count"):
            # IIF subset vs COUNT total — percent count (majority)
            return "OP: percent count"
        if iif_count == 1 and not bare_has_count and percent_denom != "sum":
            # IIF subset, no bare count but * 100 / present
            return "OP: percent count"
        if iif_count == 0:
            # * 100 / with no IIF — unusual, LLM
            if use_llm:
                return _llm_determine_op(nlq, evidence, sql, "OP: percent", client)
            return "OP: total percent"

    # subtraction
    if arith == "-":
        if _nlq_asks_compare_then_diff(nlq):
            return "OP: >,-"
        if _nlq_asks_compare_then_percent(nlq):
            return "OP: >, percent"
        # Check if any IIF branch returns a column value (semantic aggregate diff)
        if iif_count >= 2:
            returns_col = any(b["returns_column"] for b in select_info["iif_branches"])
            if returns_col:
                return "OP: -"
            return "OP: -"
        return "OP: -"

    # division (ratio)
    if arith == "/":
        if _nlq_asks_percentage(nlq):
            if use_llm:
                return _llm_determine_op(nlq, evidence, sql, "OP: ratio", client)
            return "OP: ratio"
        return "OP: ratio"

    if iif_count == 0 and bare_has_count and bare_count == 1:
        if _nlq_asks_percentage(nlq):
            return "OP: total percent"
        return "OP: total"

    if iif_count >= 1 and bare_has_count and arith is None:
        if _nlq_asks_percentage(nlq):
            return "OP: percent count"
        return "OP: percent"

    # multiple IIF with no arithmetic detected
    if iif_count >= 2 and arith is None:
        if _nlq_asks_ratio(nlq):
            return "OP: ratio"
        if _nlq_asks_percentage(nlq):
            return "OP: percent"
        return "OP: -"


    if iif_count == 0 and bare_count >= 1 and not bare_has_count:
        return None
    if iif_count == 1 and arith is None and not bare_has_count:
        returns_col = any(b.get("returns_column", False) for b in select_info["iif_branches"])
        if returns_col:
            return None
    if use_llm:
        return _llm_determine_op(nlq, evidence, sql, "OP: total", client)
    return None


# Pipeline Assembly

def _fmt_op(op_type: str, instruction: str) -> str:
    return f"{op_type} - {instruction}"


def assemble_pipeline(join_ops: list[str],
                      where_filters: list[str],
                      group_ops: list[str],
                      having_filters: list[str],
                      select_info: dict,
                      branch_filter_instrs: list[str],
                      branch_agg_instrs: list[str | None],
                      extract_instrs: list[str],
                      non_iif_agg_instrs: list[str],
                      rank_instr: str | None,
                      op_str: str | None,
                      denominator_branch_instrs: list[str] | None = None) -> str:
    """
    Assemble the format string following SQL execution order.

    Delimiter rules:
      && separates sequential operations within a trunk or branch.
      / opens a branch, | closes it.
      Branching with shared trunk:  trunk / branch1 | / branch2 | && OP
      Branching without trunk:      filter1 && filter2 && OP
      No branching:                 op1 && op2 && op3
    """
    rest_trunk = where_filters + group_ops + having_filters
    trunk_parts = join_ops + rest_trunk
    has_trunk = bool(trunk_parts)
    has_branching = select_info["has_branching"]
    iif_count = len(select_info["iif_branches"])

    # use count after last join (if joins) else original doc set.
    # when joins exist, split trunk so denominator path = join-only; numerator = full trunk + branch.
    percent_count_join_denom = (
        op_str == "OP: percent count" and has_branching and bool(join_ops)
    )

    def _build_branches():
        out = []
        for idx in range(iif_count):
            bparts = []
            if percent_count_join_denom:
                bparts.extend(rest_trunk)
            if idx < len(branch_filter_instrs):
                bparts.append(branch_filter_instrs[idx])
            if idx < len(branch_agg_instrs) and branch_agg_instrs[idx]:
                bparts.append(branch_agg_instrs[idx])
            out.append("/ " + " && ".join(bparts) + " |")
        if denominator_branch_instrs:
            out.append("/ " + " && ".join(denominator_branch_instrs) + " |")
        elif percent_count_join_denom:
            out.append("/ |")
        return out

    # Branching with shared trunk
    if has_branching and has_trunk:
        trunk_str = " && ".join(join_ops if percent_count_join_denom else trunk_parts)
        branches = _build_branches()
        fmt = trunk_str + " " + " ".join(branches)
        if op_str:
            fmt += " && " + op_str
        return fmt

    # Branching without shared trunk (split at root)
    # && / branch1 | / branch2 | denotes split at document set root
    if has_branching and not has_trunk:
        branches = _build_branches()
        fmt = " && " + " ".join(branches)
        if op_str:
            fmt += " && " + op_str
        return fmt

    # No branching: sequential pipeline 
    all_ops = list(trunk_parts)
    for idx in range(iif_count):
        if idx < len(branch_filter_instrs):
            all_ops.append(branch_filter_instrs[idx])
        if idx < len(branch_agg_instrs) and branch_agg_instrs[idx]:
            all_ops.append(branch_agg_instrs[idx])
    for a in non_iif_agg_instrs:
        all_ops.append(a)
    if rank_instr:
        all_ops.append(rank_instr)
    for e in extract_instrs:
        all_ops.append(e)
    if op_str:
        all_ops.append(op_str)
    return " && ".join(all_ops)

_RAW_TEMPLATE_PATTERNS = [
    re.compile(r'\b(?:STRFTIME|SUBSTR|JULIANDAY)\s*\(', re.I),
    re.compile(r'`'),
    re.compile(r'\b[A-Za-z]\w*\.[A-Za-z]'),
    re.compile(r'\bOR\b'),
    re.compile(r'_[:%]|__\.'),
]


def _template_is_raw(template: str) -> bool:
    """Detect if a Tier-1 template still contains raw SQL artifacts needing LLM cleanup.

    Catches: STRFTIME/SUBSTR function calls, backtick-quoted identifiers,
    alias.column dot notation (T2.col), leaked SQL OR keywords,
    unbalanced parentheses from SQL WHERE clauses, and SQL LIKE wildcard patterns.
    """
    if not template:
        return False
    if any(p.search(template) for p in _RAW_TEMPLATE_PATTERNS):
        return True
    if template.count(')') > template.count('('):
        return True
    return False


def _build_pipeline_for_select_expr(
    expr: str,
    alias_map: dict,
    nlq: str,
    evidence: str,
    sql: str,
    use_llm: bool,
    client,
) -> tuple[dict, list, list, list, list, str | None, list | None]:
    """Build per-expression pipeline parts for one SELECT expression.
    Returns (select_info, branch_filter_instrs, branch_agg_instrs, extract_instrs,
             non_iif_agg_instrs, op_str, denominator_branch_instrs)."""
    select_info = parse_select(expr, alias_map)

    branch_filter_instrs = []
    branch_agg_instrs = []
    for branch in select_info["iif_branches"]:
        cond_pred = classify_predicate(branch["condition"], alias_map)
        if use_llm and _template_is_raw(cond_pred["template"]):
            finstr = llm_translate(nlq, evidence, branch["condition"],
                                   "FILTER", cond_pred["template"], client)
        else:
            finstr = cond_pred["template"]
        branch_filter_instrs.append(_fmt_op("FILTER", finstr))

        if branch["returns_column"]:
            _, agg_col = _resolve_col(branch["true_value"], alias_map)
            if use_llm:
                ainstr = llm_semantic_instruction(
                    nlq, evidence, agg_col, "AGGREGATE",
                    f"SUM(IIF({branch['condition']}, {branch['true_value']}, 0))", client)
            else:
                ainstr = f"Sum of {_col_display(agg_col)}"
            branch_agg_instrs.append(_fmt_op("AGGREGATE", ainstr))
        else:
            branch_agg_instrs.append(None)

    non_iif_agg_instrs = []
    for agg in select_info["bare_aggregates"]:
        if agg["func"].upper() == "COUNT":
            continue
        _, agg_col = _resolve_col(agg["column"], alias_map)
        if use_llm:
            instr = llm_semantic_instruction(
                nlq, evidence, agg_col, "AGGREGATE",
                f"{agg['func']}({agg['column']})", client)
        else:
            instr = f"{agg['func'].capitalize()} of {_col_display(agg_col)}"
        non_iif_agg_instrs.append(_fmt_op("AGGREGATE", instr))

    extract_instrs = []
    for col in select_info["output_columns"]:
        if use_llm:
            instr = llm_semantic_instruction(
                nlq, evidence, col["column"], "EXTRACT",
                col["raw"], client)
        else:
            instr = f"The {_col_display(col['column'])}"
        extract_instrs.append(_fmt_op("EXTRACT", instr))

    op_str = determine_op_type(
        select_info,
        select_str=expr,
        nlq=nlq, evidence=evidence, sql=sql,
        use_llm=use_llm, client=client,
    )

    denominator_branch_instrs = None
    if op_str in ("OP: percent count", "OP: percent sum") and select_info["has_branching"]:
        if op_str == "OP: percent sum":
            sum_aggs = [a for a in select_info["bare_aggregates"] if a["func"].upper() == "SUM"]
            if sum_aggs:
                agg = sum_aggs[0]
                _, agg_col = _resolve_col(agg["column"], alias_map)
                if use_llm:
                    sinstr = llm_semantic_instruction(
                        nlq, evidence, agg_col, "AGGREGATE",
                        f"SUM({agg['column']})", client)
                else:
                    sinstr = f"Sum of {_col_display(agg_col)}"
                denominator_branch_instrs = [_fmt_op("AGGREGATE", sinstr)]

    return (
        select_info,
        branch_filter_instrs,
        branch_agg_instrs,
        extract_instrs,
        non_iif_agg_instrs,
        op_str,
        denominator_branch_instrs,
    )


def process_question(entry: dict, use_llm: bool = True, client=None) -> str:
    """Process one question entry → pipeline format string.
    Multi-answer: SELECT a, b, c → pipelines split by comma; joined with |-|."""
    sql = entry.get("SQL", "")
    nlq = entry.get("question", "")
    evidence = entry.get("evidence", "")

    if not sql:
        return ""
    if entry.get("difficulty") == "challenging":
        return ""
    if has_subquery(sql):
        return ""

    alias_map = build_alias_map(sql)
    clauses = extract_sql_clauses(sql)

    # ---- Shared trunk (JOIN, WHERE, GROUP BY, HAVING, RANK) ----
    join_ops = build_join_ops(sql)

    where_preds = parse_where(clauses.get("WHERE", ""), alias_map)
    where_instrs = []
    for pred in where_preds:
        if use_llm and _template_is_raw(pred["template"]):
            instr = llm_translate(nlq, evidence, pred["raw"], "FILTER",
                                  pred["template"], client)
        else:
            instr = pred["template"]
        where_instrs.append(_fmt_op("FILTER", instr))

    group_cols = parse_group_by(clauses.get("GROUP BY", ""), alias_map)
    group_instrs = []
    for g in group_cols:
        if use_llm and _template_is_raw(g["template"]):
            instr = llm_translate(nlq, evidence, g["raw"], "GROUP",
                                  g["template"], client)
        else:
            instr = g["template"]
        group_instrs.append(_fmt_op("GROUP", instr))

    having_preds = parse_where(clauses.get("HAVING", ""), alias_map)
    having_instrs = []
    for pred in having_preds:
        if use_llm and _template_is_raw(pred["template"]):
            instr = llm_translate(nlq, evidence, pred["raw"], "FILTER",
                                  pred["template"], client)
        else:
            instr = pred["template"]
        having_instrs.append(_fmt_op("FILTER", instr))

    rank_info = parse_order_by_limit(
        clauses.get("ORDER BY", ""), clauses.get("LIMIT", ""), alias_map)
    rank_instr = None
    if rank_info:
        if use_llm and _template_is_raw(rank_info["template"]):
            rinstr = llm_translate(nlq, evidence, rank_info["raw"],
                                   "RANK", rank_info["template"], client)
        else:
            rinstr = rank_info["template"]
        rank_instr = _fmt_op("RANK", rinstr)

    # Split SELECT on commas, one pipeline per expression
    select_body = clauses.get("SELECT", "")
    select_exprs = _split_select_on_commas(select_body)
    if not select_exprs:
        select_exprs = [select_body] if select_body.strip() else [""]
    if not select_exprs or not select_exprs[0]:
        return ""

    pipeline_fmts = []
    for expr in select_exprs:
        (
            select_info,
            branch_filter_instrs,
            branch_agg_instrs,
            extract_instrs,
            non_iif_agg_instrs,
            op_str,
            denominator_branch_instrs,
        ) = _build_pipeline_for_select_expr(
            expr, alias_map, nlq, evidence, sql, use_llm, client
        )

        fmt = assemble_pipeline(
            join_ops=join_ops,
            where_filters=where_instrs,
            group_ops=group_instrs,
            having_filters=having_instrs,
            select_info=select_info,
            branch_filter_instrs=branch_filter_instrs,
            branch_agg_instrs=branch_agg_instrs,
            extract_instrs=extract_instrs,
            non_iif_agg_instrs=non_iif_agg_instrs,
            rank_instr=rank_instr,
            op_str=op_str,
            denominator_branch_instrs=denominator_branch_instrs,
        )
        pipeline_fmts.append(fmt)

    return " |-| ".join(pipeline_fmts)


def main():
    parser = argparse.ArgumentParser(description="Build semantic pipelines from SQL")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM calls; use Tier-1 templates only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated formats without writing to file")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N questions (0 = all)")
    args = parser.parse_args()

    use_llm = not args.no_llm

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Catalog existing OP types for reference
    existing_ops = collect_existing_ops(data)
    print(f"Existing OP types found: {sorted(existing_ops)}")
    print(f"Total questions: {len(data)}")

    client = None
    if use_llm:
        client = _get_llm_client()
        if client is None:
            print("[WARN] LLM client unavailable — falling back to templates")
            use_llm = False

    entries = data[:args.limit] if args.limit > 0 else data
    updated = 0
    skipped_subquery = 0

    for idx, entry in enumerate(entries):
        qid = entry.get("question_id", idx)
        sql = entry.get("SQL", "")

        if not sql:
            continue

        fmt = process_question(entry, use_llm=use_llm, client=client)

        if not fmt:
            skipped_subquery += 1
            if args.dry_run:
                print(f"[{qid}] SKIPPED (subquery/complex)")
            entry["format"] = ""
            continue

        if args.dry_run:
            print(f"\n[{qid}] {entry.get('question', '')[:80]}")
            print(f"  SQL: {sql[:120]}...")
            print(f"  FMT: {fmt}")
        else:
            entry["format"] = fmt

        updated += 1
        print(f"Processed Question {updated}")

        if use_llm and updated % 10 == 0:
            time.sleep(0.5)

    print(f"\nProcessed: {updated} | Skipped (subquery): {skipped_subquery}")

    if not args.dry_run:
        with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Written to {QUESTIONS_PATH}")
    else:
        print("(dry-run — no file written)")


if __name__ == "__main__":
    main()
