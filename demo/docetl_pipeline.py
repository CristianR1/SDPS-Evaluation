import os
import json
import yaml
import shutil
import subprocess
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

class Operation(ABC):
    def __init__(self, instruction: str):
        self.instruction = instruction
    @abstractmethod
    def to_yaml_config(self) -> dict:
        pass
    @abstractmethod
    def modifies_docset(self) -> bool:
        pass

class FilterOperation(Operation):
    def to_yaml_config(self) -> dict:
        return {
            "name": f"filter_{id(self)}",
            "type": "filter",
            "prompt": 'Analyze the following narrative: "{{input.contents}}". If {instruction} within the narrative, respond with true'.format(instruction=self.instruction.lower()),
            "output": {"schema": {"keep": "boolean"}},
            "model": "gpt-4o-mini"
        }
    def modifies_docset(self) -> bool:
        return True

class ExtractOperation(Operation):
    def __init__(self, instruction: str, output_schema: dict = None):
        super().__init__(instruction)
        self.output_schema = output_schema or {"extracted_value": "str"}
    def to_yaml_config(self) -> dict:
        return {
            "name": f"extract_{id(self)}",
            "type": "extract",
            "prompt": 'Analyze the narrative. Extract {instruction}'.format(instruction=self.instruction.lower()),
            "document_keys": ["contents"],
            "output": {"schema": self.output_schema},
            "model": "gpt-4o-mini"
        }
    def modifies_docset(self) -> bool:
        return False

class RankOperation(Operation):
    def __init__(self, instruction: str, k: int = 1):
        super().__init__(instruction)
        self.k = k
    def to_yaml_config(self) -> dict:
        return {
            "name": f"rank_{id(self)}",
            "type": "reduce",
            "reduce_key": ["_all"],
            "prompt": 'Analyze the following narrative: "{{input.contents}}". {instruction}'.format(instruction=self.instruction.lower()),
            "output": {"schema": {"ranked_items": "list[dict]"}},
            "model": "gpt-4o-mini",
            "pass_through": True
        }
    def modifies_docset(self) -> bool:
        return True

class JoinOperation(Operation):
    def __init__(self, instruction: str, comparison_prompt: str = None):
        super().__init__(instruction)
        self.comparison_prompt = comparison_prompt or instruction
    def to_yaml_config(self) -> dict:
        return {
            "name": f"join_{id(self)}",
            "type": "resolve",
            "comparison_prompt": 'Analyze the following narrative: "{{input.contents}}". {instruction}'.format(instruction=self.instruction.lower()),
            "resolution_prompt": "Combine these matching records",
            "output": {"schema": {"joined_record": "dict"}},
            "model": "gpt-4o-mini"
        }
    def modifies_docset(self) -> bool:
        return True

class AggregateOperation(Operation):
    def to_yaml_config(self) -> dict:
        return {
            "name": f"aggregate_{id(self)}",
            "type": "reduce",
            "reduce_key": ["_all"],
            "prompt": 'Analyze the following narrative: "{{input.contents}}". {instruction}'.format(instruction=self.instruction.lower()),
            "output": {"schema": {"aggregated_result": "str"}},
            "model": "gpt-4o-mini"
        }
    def modifies_docset(self) -> bool:
        return False

class GroupOperation(Operation):
    def __init__(self, instruction: str, group_keys: list = None):
        super().__init__(instruction)
        self.group_keys = group_keys or []
    def to_yaml_config(self) -> dict:
        return {
            "name": f"group_{id(self)}",
            "type": "reduce",
            "reduce_key": self.group_keys if self.group_keys else ["group_key"],
            "prompt": 'Analyze the following narrative: "{{input.contents}}". {instruction}'.format(instruction=self.instruction.lower()),
            "output": {"schema": {"grouped_result": "dict"}},
            "model": "gpt-4o-mini"
        }
    def modifies_docset(self) -> bool:
        return True

class DocumentManager:
    def __init__(self, base_path: str, database_name: str, table_name: str):
        self.base_path = Path(base_path)
        self.table_name = table_name
        self.data_path = self.base_path / "data" / database_name / table_name
        self.interim_path = self.base_path / "interim"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interim_path.mkdir(parents=True, exist_ok=True)
    def load_documents(self, source_path: str = None) -> list:
        path = Path(source_path) if source_path else self.data_path
        documents = []
        for txt_file in sorted(path.glob("*.txt")):
            with open(txt_file, 'r', encoding='utf-8') as f:
                documents.append({"filename": txt_file.name, "contents": f.read(), "filepath": str(txt_file)})
        return documents
    def save_documents_json(self, documents: list, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2)
    def save_interim_docset(self, documents: list, operation_idx: int):
        interim_dir = self.interim_path / str(operation_idx)
        interim_dir.mkdir(parents=True, exist_ok=True)
        for doc in documents:
            filepath = doc.get("filepath", "")
            if filepath and Path(filepath).exists():
                shutil.copy(filepath, interim_dir / doc["filename"])
            else:
                with open(interim_dir / doc["filename"], 'w', encoding='utf-8') as f:
                    f.write(doc.get("contents", ""))
        return interim_dir
    def load_interim_docset(self, operation_idx: int) -> list:
        interim_dir = self.interim_path / str(operation_idx)
        return self.load_documents(str(interim_dir))
    def clear_interim(self):
        if self.interim_path.exists():
            shutil.rmtree(self.interim_path)
        self.interim_path.mkdir(parents=True, exist_ok=True)

class Pipeline:
    def __init__(self, doc_manager: DocumentManager, verbose: bool = False):
        self.doc_manager = doc_manager
        self.operations = []
        self.results = []
        self.verbose = verbose
    def log(self, message: str):
        if self.verbose:
            print(f"[DOCETL] {message}")
    def add_operation(self, operation: Operation):
        self.operations.append(operation)
    def parse_format(self, format_str: str):
        self.operations = []
        parts = [p.strip() for p in format_str.split(",")]
        for part in parts:
            if " - " in part:
                op_type, instruction = part.split(" - ", 1)
                op_type = op_type.strip().upper()
                if op_type == "FILTER":
                    self.operations.append(FilterOperation(instruction))
                elif op_type == "EXTRACT":
                    self.operations.append(ExtractOperation(instruction))
                elif op_type == "RANK":
                    self.operations.append(RankOperation(instruction))
                elif op_type == "JOIN":
                    self.operations.append(JoinOperation(instruction))
                elif op_type == "AGGREGATE":
                    self.operations.append(AggregateOperation(instruction))
                elif op_type == "GROUP":
                    self.operations.append(GroupOperation(instruction))
        self.log(f"Parsed {len(self.operations)} operations from format string")
    def clear(self):
        self.log("Clearing interim data...")
        self.doc_manager.clear_interim()
        self.log("Interim data cleared")
    def generate_yaml_config(self, output_path: str, input_json_path: str):
        self.log(f"Generating YAML config at {output_path}")
        operations_config = []
        steps = []
        for idx, op in enumerate(self.operations):
            op_config = op.to_yaml_config()
            op_config["name"] = f"op_{idx}_{type(op).__name__.lower().replace('operation', '')}"
            operations_config.append(op_config)
            step = {
                "name": op_config["name"],
                "operations": [op_config["name"]]
            }
            if idx == 0:
                step["input"] = "input_documents"
            else:
                step["input"] = f"op_{idx-1}_{type(self.operations[idx-1]).__name__.lower().replace('operation', '')}"
            steps.append(step)
        input_json_relative = Path(input_json_path).relative_to(self.doc_manager.base_path)
        config = {
            "datasets": {
                "input_documents": {
                    "type": "file",
                    "path": str(input_json_relative)
                }
            },
            "default_model": "gpt-4o-mini",
            "operations": operations_config,
            "pipeline": {
                "name": "semantic_pipeline",
                "steps": steps,
                "output": {
                    "type": "file",
                    "path": "results/docetl/docetl_result.json",
                    "intermediate_dir": "interim"
                }
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        self.log(f"YAML config generated with {len(operations_config)} operations")
        return config
    def execute(self, clear_interim: bool = False) -> list:
        if clear_interim:
            self.clear()
        self.log(f"Loading documents from {self.doc_manager.data_path}")
        documents = self.doc_manager.load_documents()
        self.log(f"Loaded {len(documents)} documents")
        self.results = [{"operation": "initial", "doc_count": len(documents)}]
        input_json = self.doc_manager.base_path / "input_documents.json"
        self.doc_manager.save_documents_json(documents, str(input_json))
        self.log(f"Saved input documents to {input_json}")
        yaml_path = self.doc_manager.base_path / "generated_pipeline.yaml"
        self.generate_yaml_config(str(yaml_path), str(input_json))
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        
        # Try to find docetl command - on Windows it might not be in PATH
        docetl_cmd = None
        # Try direct command first (most reliable)
        if shutil.which("docetl"):
            docetl_cmd = ["docetl", "run", "generated_pipeline.yaml"]
        # Try with .exe extension on Windows
        elif os.name == 'nt' and shutil.which("docetl.exe"):
            docetl_cmd = ["docetl.exe", "run", "generated_pipeline.yaml"]
        # If not found, provide helpful error
        else:
            error_msg = (
                "\n" + "="*80 + "\n"
                "❌ DocETL command not found in PATH.\n\n"
                "To install DocETL, run:\n"
                f"  {sys.executable} -m pip install docetl\n\n"
                "Or if using conda:\n"
                "  conda install -c conda-forge docetl\n\n"
                "After installation, make sure the Scripts directory is in your PATH.\n"
                "="*80 + "\n"
            )
            print(error_msg)
            raise RuntimeError("DocETL command not found. Please install it using: pip install docetl")
        
        self.log(f"Running docetl command: {' '.join(docetl_cmd)}")
        try:
            result = subprocess.run(
                docetl_cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                encoding='utf-8',
                errors='replace',
                cwd=str(self.doc_manager.base_path)
            )
            if result.stdout:
                self.log(f"DocETL stdout: {result.stdout}")
            if result.stderr:
                self.log(f"DocETL stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            self.log(f"DocETL command failed with exit code {e.returncode}")
            if e.stdout:
                self.log(f"DocETL stdout: {e.stdout}")
                print("STDOUT:", e.stdout)
            if e.stderr:
                self.log(f"DocETL stderr: {e.stderr}")
                print("STDERR:", e.stderr)
            
            # Check if docetl is not installed
            if e.stderr and "No module named docetl" in e.stderr:
                error_msg = (
                    "\n" + "="*80 + "\n"
                    "❌ DocETL is not installed in your Python environment.\n\n"
                    "To install DocETL, run:\n"
                    f"  {sys.executable} -m pip install docetl\n\n"
                    "Or if using conda:\n"
                    "  conda install -c conda-forge docetl\n"
                    "="*80 + "\n"
                )
                print(error_msg)
                raise RuntimeError("DocETL is not installed. Please install it using: pip install docetl") from e
            
            # Check for dependency version mismatch (pyrate_limiter)
            if e.stderr and ("BucketFullException" in e.stderr or "LimiterDelayException" in e.stderr or "cannot import name" in e.stderr):
                error_msg = (
                    "\n" + "="*80 + "\n"
                    "❌ DocETL dependency version mismatch detected!\n\n"
                    "The installed version of 'pyrate_limiter' is incompatible with DocETL.\n"
                    "DocETL requires an older version of pyrate_limiter.\n\n"
                    "To fix this, try:\n"
                    f"  {sys.executable} -m pip install 'pyrate_limiter<3.0.0'\n\n"
                    "Or reinstall docetl with its dependencies:\n"
                    f"  {sys.executable} -m pip install --force-reinstall docetl\n"
                    "="*80 + "\n"
                )
                print(error_msg)
                raise RuntimeError("DocETL dependency version mismatch. Try: pip install 'pyrate_limiter<3.0.0'") from e
            
            raise  
        for idx, operation in enumerate(self.operations):
            op_name = type(operation).__name__
            self.log(f"[{idx+1}/{len(self.operations)}] Processing {op_name}: {operation.instruction}")
            if operation.modifies_docset():
                self.log(f"[{idx+1}/{len(self.operations)}] Saving interim docset to folder {idx + 1}")
                self.doc_manager.save_interim_docset(documents, idx + 1)
            else:
                if idx > 0:
                    prev_dir = self.doc_manager.interim_path / str(idx)
                    curr_dir = self.doc_manager.interim_path / str(idx + 1)
                    if prev_dir.exists():
                        shutil.copytree(prev_dir, curr_dir, dirs_exist_ok=True)
                        self.log(f"[{idx+1}/{len(self.operations)}] Carried over docset to folder {idx + 1}")
                else:
                    self.doc_manager.save_interim_docset(documents, idx + 1)
                    self.log(f"[{idx+1}/{len(self.operations)}] Saving interim docset to folder {idx + 1}")
            self.results.append({
                "operation": op_name,
                "instruction": operation.instruction,
                "doc_count": len(documents),
                "modifies_docset": operation.modifies_docset()
            })
        self.log(f"Pipeline execution complete - {len(documents)} documents processed")
        return result

if __name__ == "__main__":
    format_str = "FILTER - The source of income is Fundraising, FILTER - The date received is 2019/9/14, EXTRACT - The notes of the fundraising"
    doc_manager = DocumentManager("./pipeline_data", "student_club", "income")
    pipeline = Pipeline(doc_manager, verbose=True)
    pipeline.parse_format(format_str)
    for op in pipeline.operations:
        print(f"{type(op).__name__}: {op.instruction}")
    result = pipeline.execute(clear_interim=True)
    print(result)