"""
Palimpzest pipeline aligned with lotus_pipeline: && and / split logic,
trunk+branch execution, metric tracking, correctness comparison, CSV export.
Uses palimpzest operators: sem_filter, sem_map (extract), sem_join, sem_agg, sem_topk.
"""
import palimpzest as pz
import copy
import os
import re
import json
import shutil
import sqlite3
import argparse
import time
import csv
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

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
    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        pass

    @abstractmethod
    def modifies_docset(self) -> bool:
        pass


class FilterOperation(Operation):
    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        return dataset.sem_filter(
            self.instruction,
            depends_on=["contents"],
        )

    def modifies_docset(self) -> bool:
        return True


class ExtractOperation(Operation):
    def __init__(self, instruction: str):
        super().__init__(instruction)
        self.output_cols = [
            {
                "name": "extraction",
                "type": str,
                "desc": f"Extract ONLY: {instruction}. Return just the value, nothing else.",
            }
        ]

    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        return dataset.sem_map(self.output_cols, depends_on=["contents"])

    def modifies_docset(self) -> bool:
        return False


class RankOperation(Operation):
    def __init__(self, instruction: str, k: int = 1):
        super().__init__(instruction)
        self.k = k

    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        return dataset.sem_topk(
            self.instruction,
            k=self.k,
            depends_on=["contents"],
        )

    def modifies_docset(self) -> bool:
        return True


def _join_instruction_for_palimpzest(instruction: str) -> str:
    """Use natural-language instruction as-is; questions.json uses 'one document' / 'the other document'."""
    return instruction.strip()


class JoinOperation(Operation):
    def __init__(self, instruction: str, right_dataset: pz.Dataset = None):
        super().__init__(instruction)
        self.right_dataset = right_dataset

    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        if self.right_dataset is not None:
            instr = _join_instruction_for_palimpzest(self.instruction)
            return dataset.sem_join(self.right_dataset, instr)
        return dataset

    def modifies_docset(self) -> bool:
        return True


class AggregateOperation(Operation):
    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        # sem_agg() requires col (output field spec) and agg (natural language instruction)
        return dataset.sem_agg(
            col={
                "name": "aggregate",
                "type": str,
                "desc": self.instruction,
            },
            agg=self.instruction,
            depends_on=["contents"],
        )

    def modifies_docset(self) -> bool:
        return False


class GroupOperation(Operation):
    def __init__(self, instruction: str, group_by: str = None):
        super().__init__(instruction)
        self.group_by = group_by

    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        if self.group_by:
            return dataset.group_by(self.group_by)
        return dataset.sem_partition_by(
            self.instruction,
            depends_on=["contents"],
        )

    def modifies_docset(self) -> bool:
        return True


# --- DocumentManager: multi-table like lotus, builds data_paths per table ---
class DocumentManager:
    def __init__(self, base_path: str, table_names: list):
        self.base_path = Path(base_path)
        self.table_names = table_names
        self.data_paths = []
        for table in self.table_names:
            database_name = database_tables.get(table)
            if database_name:
                self.data_paths.append(self.base_path / "data" / database_name / table)
        self.interim_path = self.base_path / "interim"
        for path in self.data_paths:
            path.mkdir(parents=True, exist_ok=True)
        self.interim_path.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, source_path: str) -> pz.Dataset:
        path = str(source_path)
        table_name = Path(source_path).name
        return pz.TextFileDataset(id=f"documents-{table_name}", path=path)

    def load_datasets(self) -> list:
        return [self.load_dataset(p) for p in self.data_paths]

    def clear_interim(self):
        if self.interim_path.exists():
            shutil.rmtree(self.interim_path)
        self.interim_path.mkdir(parents=True, exist_ok=True)


def _output_to_df(output) -> pd.DataFrame:
    """Convert palimpzest run output to DataFrame (records with attributes)."""
    try:
        return output.to_df()
    except Exception:
        records = []
        for r in output:
            row = {}
            for attr in dir(r):
                if not attr.startswith("_"):
                    try:
                        v = getattr(r, attr)
                        if not callable(v):
                            row[attr] = v
                    except Exception:
                        pass
            records.append(row)
        return pd.DataFrame(records)


def _records_to_memory_vals(records) -> list:
    """Convert trunk output records to list of dicts for MemoryDataset."""
    vals = []
    for r in records:
        vals.append({
            "filename": getattr(r, "filename", ""),
            "contents": getattr(r, "contents", ""),
            "filepath": getattr(r, "filepath", ""),
        })
    return vals


MEMORY_SCHEMA = [
    {"name": "filename", "type": str, "desc": "Filename"},
    {"name": "contents", "type": str, "desc": "Document contents"},
    {"name": "filepath", "type": str, "desc": "File path"},
]


class Pipeline:
    def __init__(self, doc_manager: DocumentManager, verbose: bool = False):
        self.doc_manager = doc_manager
        self.pipelines = []
        self.results = []
        self.verbose = verbose
        self.operations = []

    def log(self, message: str):
        if self.verbose:
            print(f"[PALIMPZEST] {message}")

    def _parse_ops_from_segment(self, segment_str: str, collect_ops: bool = True) -> list:
        """Parse a segment (comma-separated 'TYPE - instruction') into list of Operation objects.
        If collect_ops=True, appends OP:... to self.operations; use False when building path ops.
        """
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
            elif collect_ops and ": " in part:
                _, operations_str = part.split(": ", 1)
                for op in operations_str.split(","):
                    self.operations.append(op.strip())
        return operations

    @staticmethod
    def _tree_leaf_paths(n: int) -> list:
        """
        Treat n segments as a complete binary tree (index 0 = root; 2*i+1 left, 2*i+2 right).
        Return all root-to-leaf paths (each path = list of segment indices).
        """
        if n <= 0:
            return []
        if n == 1:
            return [[0]]
        leaves = [i for i in range(n) if 2 * i + 1 >= n]
        paths = []
        for leaf in leaves:
            path = []
            i = leaf
            while True:
                path.append(i)
                if i == 0:
                    break
                i = (i - 1) // 2
            path.reverse()
            paths.append(path)
        return paths

    def parse_format(self, format_str: str):
        """
        Palimpzest-friendly parse:
        - No && and no / : one pipeline (one path), executed whole.
        - && : split into separate trees; each tree expanded to paths independently.
        - /  : build a tree of segments; expand to all root-to-leaf paths (one full pipeline per path).
        So e.g. one split then two splits (4 leaves) => 4 pipelines executed.
        """
        self.pipelines = []
        self.operations = []
        tree_strings = [t.strip() for t in format_str.split("&&") if t.strip()]
        for ops_str in tree_strings:
            if not ops_str:
                continue
            segments = [s.strip() for s in ops_str.split(" / ")]
            if not segments:
                continue
            segment_ops = []
            for seg in segments:
                seg_ops = self._parse_ops_from_segment(seg, collect_ops=True)
                segment_ops.append(seg_ops)
            paths = self._tree_leaf_paths(len(segments))
            self.pipelines.append({
                "segments": segments,
                "segment_ops": segment_ops,
                "paths": paths,
            })
        total_pipelines = sum(len(p["paths"]) for p in self.pipelines)
        self.log(f"Parsed {len(self.pipelines)} tree(s), {total_pipelines} root-to-leaf path(s) (one pipeline each)")

    def clear(self):
        self.log("Clearing interim data...")
        self.doc_manager.clear_interim()
        self.log("Interim data cleared")

    def execute(self, run_mode: str = "max_quality", clear_interim: bool = False) -> list:
        """
        Execute one full palimpzest pipeline per root-to-leaf path.
        Each path is built as a single chain of operations and run() once (palimpzest executes whole).
        Returns list of DataFrames, one per path.
        """
        if clear_interim:
            self.clear()
        run_kw = {"max_quality": True} if run_mode == "max_quality" else {"min_cost": True}
        dfs_out = []
        datasets = self.doc_manager.load_datasets()
        total_docs = sum(len(list(Path(p).glob("*.txt"))) for p in self.doc_manager.data_paths)
        self.results = [{"operation": "initial", "doc_count": total_docs, "tables": len(datasets)}]

        for tree_idx, pipeline in enumerate(self.pipelines):
            segment_ops = pipeline["segment_ops"]
            paths = pipeline["paths"]

            for path_idx, path in enumerate(paths):
                path_ops = []
                for seg_idx in path:
                    for op in segment_ops[seg_idx]:
                        if type(op).__name__ == "GroupOperation":
                            continue
                        path_ops.append(copy.deepcopy(op))

                if not path_ops:
                    self.log(f"Tree {tree_idx + 1} path {path_idx + 1}: no ops, skipping")
                    continue

                join_idx = 0
                while join_idx < len(path_ops) and type(path_ops[join_idx]).__name__ == "JoinOperation":
                    join_idx += 1
                join_ops = path_ops[:join_idx]
                rest_ops = path_ops[join_idx:]

                current = None
                if not datasets:
                    self.log("No datasets to load")
                    continue
                if not join_ops:
                    current = datasets[0]
                    dataset_list = datasets[1:]
                else:
                    current = datasets[0]
                    dataset_list = datasets[1:]
                    for i, join_op in enumerate(join_ops):
                        if i >= len(dataset_list):
                            break
                        join_op.right_dataset = dataset_list[i]
                        self.log(f"Path {path_idx + 1} JoinOperation: {join_op.instruction}")
                        current = join_op.execute(current)
                        self.results.append({
                            "operation": "JoinOperation",
                            "instruction": join_op.instruction,
                            "modifies_docset": True,
                        })

                if current is None:
                    continue

                for op in rest_ops:
                    op_name = type(op).__name__
                    self.log(f"Path {path_idx + 1} {op_name}: {op.instruction}")
                    current = op.execute(current)
                    self.results.append({
                        "operation": op_name,
                        "instruction": op.instruction,
                        "modifies_docset": op.modifies_docset(),
                    })

                self.log(f"Running pipeline for tree {tree_idx + 1} path {path_idx + 1} (whole)...")
                output = current.run(**run_kw)
                df = _output_to_df(output)
                dfs_out.append(df)
                if self.results:
                    self.results[-1]["doc_count"] = len(df) if not df.empty else 0
                self.log(f"Path {path_idx + 1} complete - {len(df)} rows")

        return dfs_out


# --- Reused from lotus_pipeline: SQL and evaluation helpers ---
def extract_tables_from_sql(sql):
    if not sql or not isinstance(sql, str):
        return []
    sql_norm = sql.replace("\n", " ")
    pattern = r"(?:FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN)\s+([a-zA-Z0-9_]+)(?:\s+AS\s+\w+)?"
    matches = re.findall(pattern, sql_norm, re.IGNORECASE)
    pattern2 = r"(?:FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN)\s+`([a-zA-Z0-9_]+)`"
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
        if c == "(":
            depth += 1
            out.append(c)
            i += 1
            continue
        if c == ")":
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
        for kw in ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FROM", "JOIN"]:
            parts = kw.split()
            j = i
            for p in parts:
                while j < n and sql[j] in " \t":
                    j += 1
                if j + len(p) > n or sql[j : j + len(p)].upper() != p.upper():
                    break
                j += len(p)
            else:
                if j < n and (sql[j].isalnum() or sql[j] == "_"):
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
        while i < n and sql[i] in " \t":
            i += 1
        table_start = i
        while i < n and (sql[i].isalnum() or sql[i] == "_"):
            i += 1
        table_name = sql[table_start:i]
        if table_name.lower() not in tables_set:
            out.append(sql[table_start:i])
            while i < n and sql[i] in " \t":
                i += 1
            if i + 2 <= n and sql[i : i + 2].upper() == "AS":
                i += 2
                while i < n and sql[i] in " \t":
                    i += 1
                while i < n and (sql[i].isalnum() or sql[i] == "_"):
                    i += 1
            continue
        alias = table_name
        while i < n and sql[i] in " \t":
            i += 1
        if i + 2 <= n and sql[i : i + 2].upper() == "AS":
            i += 2
            while i < n and sql[i] in " \t":
                i += 1
            alias_start = i
            while i < n and (sql[i].isalnum() or sql[i] == "_"):
                i += 1
            alias = sql[alias_start:i]
        subq = f"(SELECT * FROM {table_name} ORDER BY ROWID LIMIT {num_documents}) AS {alias} "
        out.append(subq)
    return "".join(out)


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
    sql_results, _ = execute_sql(db_path, sql, database_name)
    ground_truth = [row[0] if len(row) == 1 else row for row in sql_results]
    print(f"\nExtracted: {final_result}")
    print(f"Ground Truth: {ground_truth}")
    return final_result, ground_truth


ALLOWED_DB_IDS = {"debit_card_specializing", "student_club"}
ALLOWED_DIFFICULTIES = {"simple", "moderate"}


def _normalize_for_compare(val) -> str:
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
    if not Path(db_path).exists():
        return []
    try:
        results, _ = execute_sql(db_path, sql, database_name)
        return [row[0] if len(row) == 1 else row for row in results]
    except Exception:
        return []


def aggregate_metrics_to_csv(rows: list, output_dir: str = "./pipeline_data/results/palimpzest") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "palimpzest_metrics.csv"
    fieldnames = [
        "question_number", "question_id", "db_id", "difficulty", "num_documents",
        "altered_sql", "execution_time_seconds", "llm_total_tokens", "llm_total_cost",
        "correct", "extracted", "ground_truth",
    ]
    if not rows:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return str(filepath)
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
    summary_path = output_path / "palimpzest_metrics_summary.csv"
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
    parser.add_argument("--num_documents", type=int, default=10)
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

            start_time = time.perf_counter()
            result = None
            try:
                result = pipeline.execute(run_mode="max_quality", clear_interim=True)
            except Exception as e:
                execution_time_seconds = time.perf_counter() - start_time
                print(f"\n[PALIMPZEST] Error on question {question_number} ({entry.get('question_id', '')}): {e}")
                import traceback
                traceback.print_exc()
                # Mark as wrong and record metrics so the set can finish
                limited_sql = limit_sql_to_num_documents(entry["SQL"], num_documents)
                ground_truth = _get_ground_truth(str(DB_PATH), database_name, limited_sql) if DB_PATH.exists() and limited_sql else []
                metrics_rows.append({
                    "question_number": question_number,
                    "question_id": entry.get("question_id", ""),
                    "db_id": entry["db_id"],
                    "difficulty": entry["difficulty"],
                    "num_documents": 0,
                    "altered_sql": limited_sql,
                    "execution_time_seconds": round(execution_time_seconds, 4),
                    "llm_total_tokens": 0,
                    "llm_total_cost": 0.0,
                    "correct": False,
                    "extracted": [],
                    "ground_truth": ground_truth,
                })
                continue

            execution_time_seconds = time.perf_counter() - start_time

            llm_tokens = 0
            llm_cost = 0.0
            num_docs_processed = pipeline.results[0]["doc_count"] if pipeline.results else 0
            limited_sql = limit_sql_to_num_documents(entry["SQL"], num_documents)

            final_result = []
            for idx, df in enumerate(result):
                try:
                    col = "contents" if "contents" in df.columns else df.columns[0] if len(df.columns) else None
                    if col is None or len(df) == 0:
                        raise ValueError
                    print(f"Dataframe {idx} contains {len(df)} documents")
                except Exception:
                    print(f"Dataframe {idx} has no documents, setting result to none")
                    final_result.append("None")
                    break

            if not final_result:
                def _doc_count(df):
                    return len(df["contents"]) if "contents" in df.columns else len(df)

                prev_was_comparison = False
                for operation in pipeline.operations:
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
                                final_result.append(str(c0 > c1))
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
                                final_result.append(str(c0 < c1))
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
                                final_result.append(f"{abs(c0 - c1)}")
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
