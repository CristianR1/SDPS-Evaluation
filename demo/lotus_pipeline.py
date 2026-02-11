import pandas as pd
import lotus
import os
import re
import json
import shutil
import sqlite3
import argparse
import time
import csv
from pathlib import Path
from lotus.models import LM
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import math

load_dotenv()
lm = LM(model="gpt-4o-mini", max_tokens=1000)
# For GroupOperation (semantic clustering), also set rm and vs, e.g.:
# from lotus.models import SentenceTransformersRM; from lotus.vector_store import FaissVS
# lotus.settings.configure(lm=lm, rm=SentenceTransformersRM(model="intfloat/e5-base-v2"), vs=FaissVS())
lotus.settings.configure(lm=lm)

database_tables = {
    "frpm": "california_schools",
    "satscores": "california_schools",
    "schools": "california_schools",

    "cards": "card_games",
    "foreign_data": "card_games",
    "legalities": "card_games",
    "sets": "card_games",
    "set_translations": "card_games",
    "ruling": "card_games",

    "customers": "debit_card_specializing",
    "gasstations": "debit_card_specializing",
    "products": "debit_card_specializing",
    "transactions_1k": "debit_card_specializing",
    "yearmonth": "debit_card_specializing",

    "account": "financial",
    "card": "financial",
    "client": "financial",
    "disp": "financial",
    "district": "financial",
    "loan": "financial",
    "order": "financial",
    "trans": "financial",

    "circuits": "formula_1",
    "constructors": "formula_1",
    "drivers": "formula_1",
    "seasons": "formula_1",
    "races": "formula_1",
    "constructorresults": "formula_1",
    "constructorstandings": "formula_1",
    "driverstandings": "formula_1",
    "laptimes": "formula_1",
    "pitstops": "formula_1",
    "qualifying": "formula_1",
    "status": "formula_1",
    "results": "formula_1",

    "player_attributes": "european_football_2",
    "player": "european_football_2",
    "league": "european_football_2",
    "country": "european_football_2",
    "team": "european_football_2",
    "team_attributes": "european_football_2",
    "match": "european_football_2",

    "examination": "thrombosis_prediction",
    "patient": "thrombosis_prediction",
    "laboratory": "thrombosis_prediction",

    "atom": "toxicology",
    "bond": "toxicology",
    "connected": "toxicology",
    "molecule": "toxicology",

    "event": "student_club",
    "major": "student_club",
    "zip_code": "student_club",
    "attendance": "student_club",
    "budget": "student_club",
    "expense": "student_club",
    "income": "student_club",
    "member": "student_club",

    "alignment": "superhero",
    "attribute": "superhero",
    "colour": "superhero",
    "gender": "superhero",
    "publisher": "superhero",
    "race": "superhero",
    "superhero": "superhero",
    "hero_attribute": "superhero",
    "superpower": "superhero",
    "hero_power": "superhero",

    "badges": "codebase_community",
    "comments": "codebase_community",
    "posthistory": "codebase_community",
    "postlinks": "codebase_community",
    "posts": "codebase_community",
    "tags": "codebase_community",
    "users": "codebase_community",
    "votes": "codebase_community",
}

class Operation(ABC):
    def __init__(self, instruction: str):
        self.instruction = instruction
    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    @abstractmethod
    def modifies_docset(self) -> bool:
        pass

class FilterOperation(Operation):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.sem_filter("{contents} includes information that indicates {instruction}".format(contents="{contents}", instruction=self.instruction))
        return result
    def modifies_docset(self) -> bool:
        return True

class ExtractOperation(Operation):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        output_cols = {"extraction": "Extract ONLY: {instruction}. Return just the value, nothing else.".format(instruction=self.instruction)}
        result = df.sem_extract(["contents"], output_cols, extract_quotes=False)
        return result
    def modifies_docset(self) -> bool:
        return False

class RankOperation(Operation):
    def __init__(self, instruction: str, k: int = 1):
        super().__init__(instruction)
        self.k = k
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.sem_topk("In {contents}, {instruction}".format(contents = "{contents}", instruction=self.instruction), self.k)
        return result
    def modifies_docset(self) -> bool:
        return True

def _content_columns(df: pd.DataFrame) -> list:
    """Return list of column names that hold document content (contents, contents:left, contents:right, etc.)."""
    cols = []
    if "contents" in df.columns:
        cols.append("contents")
    for c in df.columns:
        if isinstance(c, str) and c != "contents" and "contents" in c.lower():
            cols.append(c)
    return cols


def normalize_joined_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    After a lotus join, the DataFrame has contents:left and contents:right instead of contents.
    Merge those into a single 'contents' column so downstream Filter/Extract/Rank/Aggregate work.
    In a join chain, merge all content-like columns (contents:left, contents:right, etc.) into one.
    Also set canonical filename/filepath from :left so interim save and row identity work.
    """
    if df is None or df.empty:
        return df
    # Work on a copy with a clean RangeIndex to avoid alignment/reindex issues
    df = df.copy().reset_index(drop=True)
    if "contents" in df.columns:
        return df
    content_cols = _content_columns(df)
    if not content_cols:
        return df
    # Build a single 'contents' series by concatenating all content-like columns
    base_col = content_cols[0]
    series = df[base_col].astype(str).reset_index(drop=True)
    for c in content_cols[1:]:
        if c in df.columns:
            add = df[c].astype(str).reset_index(drop=True)
            series = series + "\n\n" + add
    df["contents"] = series
    # Canonical filename/filepath from left side so interim save and identity work
    if "filename" not in df.columns and "filename:left" in df.columns:
        df["filename"] = df["filename:left"].astype(str)
    if "filepath" not in df.columns and "filepath:left" in df.columns:
        df["filepath"] = df["filepath:left"].astype(str)
    # Drop prior right-side columns to avoid column collisions in subsequent joins
    right_cols = [c for c in df.columns if isinstance(c, str) and c.endswith(":right")]
    if right_cols:
        df = df.drop(columns=right_cols)
    return df


def build_lotus_join_instruction(instruction: str, left_df: pd.DataFrame, right_df: pd.DataFrame) -> str:
    """
    questions.json uses semantic instructions: "one document" / "the other document" (no left/right).
    Lotus maps to sem_join by labeling inputs as First/Second so the instruction stays role-based.
    After a join, left_df may have contents:left/contents:right; we treat a single 'contents' or first content column.
    """
    left_cols = _content_columns(left_df)
    right_cols = _content_columns(right_df)
    if left_cols and right_cols:
        return "First document: {contents:left}. Second document: {contents:right}. " + instruction.strip()
    return instruction

class JoinOperation(Operation):
    def __init__(self, instruction: str, right_df: pd.DataFrame = None):
        super().__init__(instruction)
        self.right_df = right_df
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.right_df is not None:
            join_instruction = build_lotus_join_instruction(self.instruction, df, self.right_df)
            print(f"Well Formed Instruction: {join_instruction}")
            result = df.sem_join(self.right_df, join_instruction)
        else:
            result = df.copy()
        return result
    def modifies_docset(self) -> bool:
        return True

class AggregateOperation(Operation):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.sem_agg("{instruction} within {contents}".format(instruction=self.instruction,contents = "{contents}"))
        return result
    def modifies_docset(self) -> bool:
        return False

class GroupOperation(Operation):
    """
    Group documents by semantic similarity: index the content column for similarity retrieval,
    then cluster so that documents with similar conditions are binarily separated (two clusters).
    Requires lotus.settings to have rm (e.g. SentenceTransformersRM) and vs (e.g. FaissVS) configured.
    """
    # Default index dir for semantic grouping (contents column)
    GROUP_INDEX_DIR = "lotus_group_contents_index"

    def __init__(self, instruction: str, group_col: str = None):
        super().__init__(instruction)
        self.group_col = group_col

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.group_col and self.group_col in df.columns:
            return df.groupby(self.group_col).apply(lambda x: x)
        # Semantic grouping: use "contents" (or first text-like column), index then cluster
        content_col = "contents" if "contents" in df.columns else None
        if content_col is None:
            for c in df.columns:
                if df[c].dtype == object or str(df[c].dtype).startswith("string"):
                    content_col = c
                    break
        if content_col is None:
            raise ValueError("GroupOperation: no 'contents' or text column found in DataFrame")
        if len(df) < 1:
            return df.copy()
        try:
            rm, vs = lotus.settings.rm, lotus.settings.vs
        except Exception:
            rm, vs = None, None
        if rm is None or vs is None:
            raise ValueError(
                "GroupOperation requires a retrieval model (RM) and vector store (VS). "
                "Configure with e.g. lotus.settings.configure(lm=..., rm=SentenceTransformersRM(...), vs=FaissVS())"
            )
        n_clusters = min(2, max(1, len(df)))
        index_dir = self.GROUP_INDEX_DIR
        if df.attrs.get("index_dirs", {}).get(content_col) != index_dir:
            df = df.sem_index(content_col, index_dir)
        return df.sem_cluster_by(content_col, n_clusters)
    def modifies_docset(self) -> bool:
        return True

class DocumentManager:
    def __init__(self, base_path: str, table_names: list):
        self.base_path = Path(base_path)
        self.table_names = table_names
        self.data_paths = []
        for table in self.table_names:
            database_name = database_tables.get(table)
            self.data_paths.append(self.base_path / "data" / database_name / table)
        self.interim_path = self.base_path / "interim"
        for path in self.data_paths:
            path.mkdir(parents=True, exist_ok=True)
        self.interim_path.mkdir(parents=True, exist_ok=True)
    def load_documents(self, source_path: str = None) -> pd.DataFrame:
        path = Path(source_path) if source_path else self.data_paths[0]
        documents = []
        for txt_file in sorted(path.glob("*.txt")):
            with open(txt_file, 'r', encoding='utf-8') as f:
                documents.append({"filename": txt_file.name, "contents": f.read(), "filepath": str(txt_file)})
        return pd.DataFrame(documents)
    def save_interim_docset(self, df: pd.DataFrame, operation_idx: int):
        interim_dir = self.interim_path / str(operation_idx)
        interim_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            src_path = Path(row.get("filepath", ""))
            if src_path.exists():
                shutil.copy(src_path, interim_dir / row["filename"])
            else:
                with open(interim_dir / row["filename"], 'w', encoding='utf-8') as f:
                    f.write(row.get("contents", ""))
        return interim_dir
    def load_interim_docset(self, operation_idx: int) -> pd.DataFrame:
        interim_dir = self.interim_path / str(operation_idx)
        return self.load_documents(str(interim_dir))
    def clear_interim(self):
        if self.interim_path.exists():
            shutil.rmtree(self.interim_path)
        self.interim_path.mkdir(parents=True, exist_ok=True)

class Pipeline:
    def __init__(self, doc_manager: DocumentManager, verbose: bool = False):
        self.doc_manager = doc_manager
        self.pipelines = []
        self.results = []
        self.verbose = verbose
        self.operations = []
    def log(self, message: str):
        if self.verbose:
            print(f"[LOTUS] {message}")

    def _parse_ops_from_segment(self, segment_str: str) -> list:
        """Parse a single segment string (comma-separated 'TYPE - instruction') into a list of Operation objects."""
        operations = []
        if not segment_str or not isinstance(segment_str, str):
            return operations
        parts = [p.strip() for p in segment_str.split(",") if p.strip()]
        for part in parts:
            if " - " in part:
                op_type, instruction = part.split(" - ", 1)
                op_type = op_type.strip().upper()
                instruction = instruction.strip()
                if op_type == "FILTER":
                    operations.append(FilterOperation(instruction))
                elif op_type == "EXTRACT":
                    operations.append(ExtractOperation(instruction))
                elif op_type == "RANK":
                    operations.append(RankOperation(instruction))
                elif op_type in ("JOIN", "LEFT JOIN", "RIGHT JOIN"):
                    operations.append(JoinOperation(instruction))
                elif op_type == "AGGREGATE":
                    operations.append(AggregateOperation(instruction))
                elif op_type == "GROUP":
                    operations.append(GroupOperation(instruction))
            elif ": " in part:
                op_type, operations_str = part.split(": ", 1)
                for op in operations_str.split(","):
                    self.operations.append(op.strip())
        return operations

    def parse_format(self, format_str: str):
        """
        Parse format string with tree-like splits.
        - '&&' separates independent pipelines.
        - ' / ' (space-slash-space) denotes a split: segment before first ' / ' is the trunk;
          each segment between consecutive ' / ' is a branch that runs on the trunk's output.
        - Empty segments (e.g. ' /  / ') are treated as branches with no ops (trunk result passed through).
        - Plain '/' inside instructions (e.g. dates 2012/8/24) is preserved.
        """
        self.pipelines = []
        pipeline_strings = [p.strip() for p in format_str.split("&&") if p.strip()]
        total_ops_parsed = 0
        for ops_str in pipeline_strings:
            if not ops_str:
                continue
            segments = [s.strip() for s in ops_str.split(" / ")]
            trunk_str = segments[0] if segments else ""
            branch_strs = segments[1:] if len(segments) > 1 else []
            trunk_ops = self._parse_ops_from_segment(trunk_str)
            total_ops_parsed += len(trunk_ops)
            branches = []
            for bstr in branch_strs:
                branch_ops = self._parse_ops_from_segment(bstr)
                total_ops_parsed += len(branch_ops)
                branches.append(branch_ops)
            self.pipelines.append({"trunk": trunk_ops, "branches": branches})
        self.log(f"Parsed {total_ops_parsed} distinct operations from format string")
        self.log(f"Parsed {len(self.pipelines)} pipeline(s) with trunk+branch structure")
    def clear(self):
        self.log("Clearing interim data...")
        self.doc_manager.clear_interim()
        self.log("Interim data cleared")
    def _run_ops_on_df(self, df: pd.DataFrame, operations: list, join_idx: int, base_step: int,
                       table_dfs: list = None, is_trunk_join_phase: bool = False) -> tuple:
        """Run a list of operations on df. For join phase, table_dfs and is_trunk_join_phase are used."""
        if is_trunk_join_phase and table_dfs is not None:
            for idx, op in enumerate(operations):
                if len(table_dfs) < 2:
                    break
                op_name = type(op).__name__
                self.log(f"Executing {op_name}: {op.instruction}")
                try:
                    op.right_df = table_dfs[1]
                    df = op.execute(table_dfs[0])
                    if hasattr(df, 'to_pandas'):
                        df = df.to_pandas()
                    df = normalize_joined_df(df)
                    table_dfs[:] = [df] + table_dfs[2:]
                    self.log(f"{op_name} complete - {len(df)} documents in joined result")
                    if len(df) == 0:
                        return df, True
                    self.results.append({"operation": op_name, "instruction": op.instruction, "doc_count": len(df), "modifies_docset": True})
                except Exception as e:
                    # Fail-safe: if a join errors out, mark this pipeline as failed and move on
                    self.log(f"Join operation failed with error: {e}. Marking pipeline as failed.")
                    return pd.DataFrame(), True
            out = table_dfs[0] if table_dfs else pd.DataFrame()
            return (normalize_joined_df(out) if not out.empty else out), False
        for idx, operation in enumerate(operations):
            op_name = type(operation).__name__
            try:
                if "contents" in df.columns:
                    print(f"{len(df['contents'])} documents remaining ...")
                    if len(df['contents']) == 0 and idx != 0:
                        raise ValueError("No documents remaining")
            except Exception:
                print(f"Exiting early, pipeline failure on operation {idx}")
                break
            self.log(f"[{idx+1}/{len(operations)}] Executing {op_name}: {operation.instruction}")
            df = operation.execute(df)
            if hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            self.log(f"[{idx+1}/{len(operations)}] {op_name} complete - {len(df)} documents remaining")
            step = base_step + idx + 1
            if operation.modifies_docset():
                self.log(f"Saving interim docset to folder {step}")
                self.doc_manager.save_interim_docset(df, step)
            else:
                self.log(f"Operation Result:")
                for col in df.columns:
                    if col not in ["filename", "contents", "filepath"]:
                        self.log(f"  Column '{col}':")
                        for i, val in enumerate(df[col].values):
                            self.log(f"    [{i}] {val}")
                if base_step + idx > 0:
                    prev_dir = self.doc_manager.interim_path / str(base_step + idx)
                    curr_dir = self.doc_manager.interim_path / str(step)
                    if prev_dir.exists():
                        shutil.copytree(prev_dir, curr_dir, dirs_exist_ok=True)
                        self.log(f"Carried over docset to folder {step}")
                else:
                    self.doc_manager.save_interim_docset(df, step)
                    self.log(f"Saving interim docset to folder {step}")
            self.results.append({
                "operation": op_name,
                "instruction": operation.instruction,
                "doc_count": len(df),
                "modifies_docset": operation.modifies_docset()
            })
        return df, False

    def execute(self, clear_interim: bool = False) -> list:
        if clear_interim:
            self.clear()
        dfs = []
        for pipeline in self.pipelines:
            trunk_ops = [op for op in pipeline["trunk"] if type(op).__name__ != "GroupOperation"]
            branches = [[op for op in b if type(op).__name__ != "GroupOperation"] for b in pipeline["branches"]]
            table_dfs = []
            for path in self.doc_manager.data_paths:
                self.log(f"Loading documents from {path}")
                df = self.doc_manager.load_documents(str(path))
                table_dfs.append(df)
                self.log(f"Loaded {len(df)} documents from {path}")
            self.results = [{"operation": "initial", "doc_count": sum(len(d) for d in table_dfs), "tables": len(table_dfs)}]
            join_idx = 0
            while join_idx < len(trunk_ops) and type(trunk_ops[join_idx]).__name__ == 'JoinOperation':
                join_idx += 1
            join_ops = trunk_ops[:join_idx]
            rest_trunk_ops = trunk_ops[join_idx:]
            ext = False
            if join_ops:
                df_trunk, ext = self._run_ops_on_df(None, join_ops, join_idx, 0, table_dfs=table_dfs, is_trunk_join_phase=True)
            else:
                df_trunk = table_dfs[0] if table_dfs else pd.DataFrame()
            if ext:
                self.log("Exiting early, pipeline failure on join phase")
                continue
            if len(df_trunk) > 0:
                self.log(f"DataFrame columns: {list(df_trunk.columns)}")
                self.log(f"First row keys: {list(df_trunk.iloc[0].keys())}")
            if rest_trunk_ops:
                df_trunk, ext = self._run_ops_on_df(df_trunk, rest_trunk_ops, join_idx, join_idx, is_trunk_join_phase=False)
                if ext:
                    continue
            if not branches:
                dfs.append(df_trunk)
                self.log(f"Pipeline execution complete (no branches) - {len(df_trunk)} documents in final result")
                continue
            for branch_idx, branch_ops in enumerate(branches):
                df_branch = df_trunk.copy()
                if not branch_ops:
                    dfs.append(df_branch)
                    self.log(f"Branch {branch_idx + 1} (empty) - passed through trunk result, {len(df_branch)} documents")
                    continue
                self.log(f"Branch {branch_idx + 1}/{len(branches)} starting from trunk state ({len(df_branch)} documents)")
                step_base = join_idx + len(rest_trunk_ops) + branch_idx * 100
                df_branch, ext = self._run_ops_on_df(df_branch, branch_ops, join_idx, step_base, is_trunk_join_phase=False)
                if not ext:
                    dfs.append(df_branch)
                    self.log(f"Branch {branch_idx + 1} complete - {len(df_branch)} documents in result")
        return dfs

def extract_tables_from_sql(sql):
    if not sql or not isinstance(sql, str):
        return []
    sql_norm = sql.replace('\n', ' ')
    pattern = r'(?:FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN)\s+([a-zA-Z0-9_]+)(?:\s+AS\s+\w+)?'
    matches = re.findall(pattern, sql_norm, re.IGNORECASE)
    pattern2 = r'(?:FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN)\s+`([a-zA-Z0-9_]+)`'
    matches.extend(re.findall(pattern2, sql_norm))
    seen = set()
    unique = []
    for m in matches:
        m_lower = m.lower()
        if m_lower not in seen:
            seen.add(m_lower)
            unique.append(m_lower)
    return unique

def limit_sql_to_num_documents(sql: str, num_documents: int) -> str:
    tables = extract_tables_from_sql(sql)
    tables = [t for t in tables if database_tables.get(t)]
    if not tables or num_documents <= 0:
        return sql
    tables_set = {t.lower() for t in tables}
    n = len(sql)
    depth = 0
    out = []
    i = 0
    while i < n:
        c = sql[i]
        if c == '(':
            depth += 1
            out.append(c)
            i += 1
            continue
        if c == ')':
            depth -= 1
            out.append(c)
            i += 1
            continue
        if depth > 0:
            out.append(c)
            i += 1
            continue
        found_kw = False
        kw_end = i
        for kw in ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FROM', 'JOIN']:
            parts = kw.split()
            j = i
            for p in parts:
                while j < n and sql[j] in ' \t':
                    j += 1
                if j + len(p) > n or sql[j:j+len(p)].upper() != p.upper():
                    break
                j += len(p)
            else:
                if j < n and (sql[j].isalnum() or sql[j] == '_'):
                    continue
                found_kw = True
                kw_end = j
                break
        if not found_kw:
            out.append(c)
            i += 1
            continue
        out.append(sql[i:kw_end])
        i = kw_end
        while i < n and sql[i] in ' \t':
            i += 1
        table_start = i
        while i < n and (sql[i].isalnum() or sql[i] == '_'):
            i += 1
        table_name = sql[table_start:i]
        if table_name.lower() not in tables_set:
            out.append(sql[table_start:i])
            while i < n and sql[i] in ' \t':
                i += 1
            if i + 2 <= n and sql[i:i+2].upper() == 'AS':
                i += 2
                while i < n and sql[i] in ' \t':
                    i += 1
                while i < n and (sql[i].isalnum() or sql[i] == '_'):
                    i += 1
            continue
        alias = table_name
        while i < n and sql[i] in ' \t':
            i += 1
        if i + 2 <= n and sql[i:i+2].upper() == 'AS':
            i += 2
            while i < n and sql[i] in ' \t':
                i += 1
            alias_start = i
            while i < n and (sql[i].isalnum() or sql[i] == '_'):
                i += 1
            alias = sql[alias_start:i]
        subq = f"(SELECT * FROM {table_name} ORDER BY ROWID LIMIT {num_documents}) AS {alias} "
        out.append(subq)
    return ''.join(out)

def execute_sql(db_path, sql: str, database_name: str):
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path / database_name / f"{database_name}.sqlite")
    cursor = conn.cursor()
    print(sql)
    cursor.execute(sql)
    results = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    conn.close()
    return results, columns

def validate_result(final_result: list, db_path: str, database_name: str, sql: str):
    sql_results, sql_columns = execute_sql(db_path, sql, database_name)
    ground_truth = [row[0] if len(row) == 1 else row for row in sql_results]
    print(f"\nExtracted: {final_result}")
    print(f"Ground Truth: {ground_truth}")
    return final_result, ground_truth


# --- Pipeline run configuration: only simple/moderate, debit_card + student_club, no GROUP ---
ALLOWED_DB_IDS = {"debit_card_specializing", "student_club"}
ALLOWED_DIFFICULTIES = {"simple", "moderate"}


def _normalize_for_compare(val) -> str:
    """Strip and clean to words/numerics only, lowercase, for comparison."""
    if val is None:
        return ""
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        parts = []
        for v in val:
            parts.extend(re.findall(r"[a-z0-9]+", str(v).lower()))
        return " ".join(sorted(parts))
    return " ".join(sorted(re.findall(r"[a-z0-9]+", str(val).lower())))


def _answers_match(extracted: list, ground_truth: list) -> bool:
    """True if stripped/cleaned words and numerics in extracted match ground_truth.
    Also robust to numeric formatting: if all ground-truth numbers appear (within tolerance)
    somewhere in the extracted answer, it counts as correct.
    """
    norm_ex = _normalize_for_compare(extracted)
    norm_gt = _normalize_for_compare(ground_truth)
    if norm_ex == norm_gt:
        return True

    # Numeric-tolerant comparison
    def _extract_numbers(val) -> list[float]:
        txt = str(val)
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
        out = []
        for n in nums:
            try:
                out.append(float(n))
            except Exception:
                continue
        return out

    ex_nums = _extract_numbers(extracted)
    gt_nums = _extract_numbers(ground_truth)
    if gt_nums and ex_nums:
        tol = 1e-2
        all_present = True
        for g in gt_nums:
            if not any(abs(e - g) <= tol for e in ex_nums):
                all_present = False
                break
        if all_present:
            return True
    return False


def _get_ground_truth(db_path: str, database_name: str, sql: str):
    """Return ground truth list from DB; empty list if DB missing or error."""
    if not Path(db_path).exists():
        return []
    try:
        results, _ = execute_sql(db_path, sql, database_name)
        return [row[0] if len(row) == 1 else row for row in results]
    except Exception:
        return []


def aggregate_metrics_to_csv(rows: list, output_dir: str = "./pipeline_data/results/lotus") -> str:
    """Write per-question metrics and summary to CSV; return path to written file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "lotus_metrics.csv"
    if not rows:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["question_number", "question_id", "db_id", "difficulty", "num_documents",
                        "altered_sql", "execution_time_seconds", "llm_total_tokens", "llm_total_cost",
                        "correct", "extracted", "ground_truth"])
        return str(filepath)
    fieldnames = ["question_number", "question_id", "db_id", "difficulty", "num_documents",
                 "altered_sql", "execution_time_seconds", "llm_total_tokens", "llm_total_cost",
                 "correct", "extracted", "ground_truth"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_export = {k: r.get(k) for k in fieldnames}
            row_export["extracted"] = str(r.get("extracted", ""))[:500]
            row_export["ground_truth"] = str(r.get("ground_truth", ""))[:500]
            writer.writerow(row_export)
    total_time = sum(r.get("execution_time_seconds") or 0 for r in rows)
    total_tokens = sum(r.get("llm_total_tokens") or 0 for r in rows)
    total_cost = sum(r.get("llm_total_cost") or 0 for r in rows)
    num_correct = sum(1 for r in rows if r.get("correct") is True)
    summary_path = output_path / "lotus_metrics_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total_questions", len(rows)])
        w.writerow(["total_execution_time_seconds", round(total_time, 4)])
        w.writerow(["total_llm_tokens", total_tokens])
        w.writerow(["total_llm_cost", round(total_cost, 6)])
        w.writerow(["num_correct", num_correct])
        w.writerow(["accuracy", round(num_correct / len(rows), 4) if rows else 0])
    return str(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_documents", type=int, default=10, help="Limit each table in SQL to this many rows (ORDER BY ROWID) to align with document set size")
    args = parser.parse_args()
    num_documents = args.num_documents

    MINIDEV_PATH = Path(__file__).parent.parent.parent / "DGS" / "data" / "MINIDEV"
    DB_PATH = MINIDEV_PATH / "dev_databases"

    with open("questions.json", encoding="utf-8") as f:
        d = json.load(f)
    if not d:
        print("Failed to load question sequence")
    else:
        database_order = list(dict.fromkeys(entry["db_id"] for entry in d))
        difficulty_order = ["simple", "moderate", "challenging"]
        ordered_entries = []
        for db_id in database_order:
            for difficulty in difficulty_order:
                for entry in d:
                    if entry["db_id"] == db_id and entry["difficulty"] == difficulty:
                        ordered_entries.append(entry)

        # Only simple/moderate, only debit_card_specializing and student_club, skip any format containing GROUP
        filtered_entries = [
            e for e in ordered_entries
            if e["db_id"] in ALLOWED_DB_IDS
            and e["difficulty"] in ALLOWED_DIFFICULTIES
            and "GROUP -" not in (e.get("format") or "")
        ]

        metrics_rows = []
        for question_number, entry in enumerate(filtered_entries, start=1):
            sequence = entry["format"]
            tables_list = entry["tables"].split(", ")
            database_name = database_tables.get(tables_list[0])
            if not database_name:
                continue
            doc_manager = DocumentManager("./pipeline_data", tables_list)
            pipeline = Pipeline(doc_manager, verbose=True)
            pipeline.parse_format(sequence)

            lotus.settings.lm.reset_stats()
            start_time = time.perf_counter()
            try:
                result = pipeline.execute(clear_interim=True)
            finally:
                execution_time_seconds = time.perf_counter() - start_time

            lm_stats = lotus.settings.lm.stats
            llm_tokens = lm_stats.physical_usage.total_tokens
            llm_cost = lm_stats.physical_usage.total_cost
            num_docs_processed = pipeline.results[0]["doc_count"] if pipeline.results else 0

            limited_sql = limit_sql_to_num_documents(entry["SQL"], num_documents)

            final_result = []
            for idx, df in enumerate(result):
                try:
                    print(f"Dataframe {idx} contains {len(df['contents'])} documents")
                    if len(df["contents"]) == 0:
                        raise ValueError
                except Exception:
                    print(f"Dataframe {idx} has no documents, setting result to none")
                    final_result.append("None")
                    break

            if not final_result:
                def _doc_count(df):
                    return len(df["contents"]) if "contents" in df.columns else len(df)

                prev_was_comparison = False
                for op_idx, operation in enumerate(pipeline.operations):
                    print(operation)
                    if operation == "ratio":
                        try:
                            if len(result) == 2:
                                c0, c1 = _doc_count(result[0]), _doc_count(result[1])
                                final_result.append(f"{c0}:{c1}")
                            else:
                                raise ValueError("ratio requires exactly two results")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline ratio operation: {e}")
                        prev_was_comparison = False
                    elif operation == "total":
                        try:
                            if len(result) == 1:
                                final_result.append(str(_doc_count(result[0])))
                            else:
                                raise ValueError("total requires exactly one result")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline total operation: {e}")
                        prev_was_comparison = False
                    elif operation == "percent reverse":
                        try:
                            if len(result) == 2:
                                c0, c1 = _doc_count(result[0]), _doc_count(result[1])
                                percent = (c1 / c0) * 100 if c0 else 0
                                final_result.append(f"{percent}")
                            else:
                                raise ValueError("percent reverse requires exactly two results")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline percent reverse operation: {e}")
                        prev_was_comparison = False
                    elif operation == "percent forward":
                        try:
                            if len(result) == 2:
                                c0, c1 = _doc_count(result[0]), _doc_count(result[1])
                                percent = (c0 / c1) * 100 if c1 else 0
                                final_result.append(f"{percent}")
                            else:
                                raise ValueError("percent forward requires exactly two results")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline percent forward operation: {e}")
                        prev_was_comparison = False
                    elif operation == "total percent":
                        continue
                    elif operation == ">":
                        try:
                            if len(result) == 2:
                                c0, c1 = _doc_count(result[0]), _doc_count(result[1])
                                comparison = c0 > c1
                                final_result.append(str(comparison))
                                prev_was_comparison = True
                            else:
                                raise ValueError("> requires exactly two results")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline > operation: {e}")
                            prev_was_comparison = False
                    elif operation == "<":
                        try:
                            if len(result) == 2:
                                c0, c1 = _doc_count(result[0]), _doc_count(result[1])
                                comparison = c0 < c1
                                final_result.append(str(comparison))
                                prev_was_comparison = True
                            else:
                                raise ValueError("< requires exactly two results")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline < operation: {e}")
                            prev_was_comparison = False
                    elif operation == "bool":
                        try:
                            if len(result) >= 1:
                                has_docs = _doc_count(result[0]) != 0
                                final_result.append("True" if has_docs else "False")
                                prev_was_comparison = True
                            else:
                                raise ValueError("bool requires at least one result")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline bool operation: {e}")
                            prev_was_comparison = False
                    elif operation == "-":
                        try:
                            if prev_was_comparison and len(result) == 2:
                                c0, c1 = _doc_count(result[0]), _doc_count(result[1])
                                difference = abs(c0 - c1)
                                final_result.append(f"{difference}")
                            elif not prev_was_comparison:
                                pass
                            else:
                                raise ValueError("- requires exactly two results")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline - operation: {e}")
                        prev_was_comparison = False
                    else:
                        prev_was_comparison = False

                for df_item in result:
                    if "extraction" in df_item.columns:
                        final_result.append(df_item["extraction"])

            ground_truth = _get_ground_truth(str(DB_PATH), database_name, limited_sql) if DB_PATH.exists() and limited_sql else []
            if DB_PATH.exists() and limited_sql:
                print(f"\nExtracted: {final_result}")
                print(f"Ground Truth: {ground_truth}")
                # Always compute correctness when DB is available; [] and None both normalize to empty
                correct = _answers_match(final_result, ground_truth)
            else:
                correct = None

            metrics_rows.append({
                "question_number": question_number,
                "question_id": entry.get("question_id", ""),
                "db_id": entry["db_id"],
                "difficulty": entry["difficulty"],
                "num_documents": num_docs_processed,
                "altered_sql": limited_sql,
                "execution_time_seconds": round(execution_time_seconds, 4),
                "llm_total_tokens": llm_tokens,
                "llm_total_cost": round(llm_cost, 6),
                "correct": correct,
                "extracted": final_result,
                "ground_truth": ground_truth,
            })

        out_path = aggregate_metrics_to_csv(metrics_rows)
        print(f"\nMetrics written to {out_path}")
        if metrics_rows:
            lotus.settings.lm.print_total_usage()