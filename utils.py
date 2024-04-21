import sys, subprocess, os, shutil
import string
import random
import re
import ast
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from functools import wraps


def timeout(time_limit):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=time_limit)
                except TimeoutError:
                    executor.shutdown(wait=False)
                    raise TimeoutError(
                        f"{func.__name__} exceeded timeout of {time_limit} seconds"
                    )
        return wrapper
    return decorator


def get_object_names_from_code(code):
    class TopLevelNameCollector(ast.NodeVisitor):
        def __init__(self):
            self.names = []

        def visit_FunctionDef(self, node):
            if isinstance(node.parent, ast.Module):
                self.names.append(node.name)

        def visit_ClassDef(self, node):
            self.names.append(node.name)

    def set_parent_nodes(node, parent=None):
        for child in ast.iter_child_nodes(node):
            child.parent = node
            set_parent_nodes(child, node)

    tree = ast.parse(code)
    set_parent_nodes(tree)

    collector = TopLevelNameCollector()
    collector.visit(tree)
    return set(collector.names)


def delete_object_from_code(func_name, code):
    class FunctionDeleter(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == func_name:
                return None
            return node
    tree = ast.parse(code)
    modified_tree = FunctionDeleter().visit(tree)
    modified_code = ast.unparse(modified_tree)
    return modified_code


def strip_ansi_escape_sequences(s):
    ansi_escape_regex = re.compile(r"\x1b\[([0-9A-Za-z]+)(;[0-9]+)*m")
    return ansi_escape_regex.sub("", s)


def extract_code(text):
    pattern = r"```python\s*(.*?)\s*```|'''(.*?)'''|\"\"\"(.*?)\"\"\""
    matches = re.findall(pattern, text, re.DOTALL)

    code_blocks = []
    for match in matches:
        code_block = "".join(match).strip()
        if code_block:
            code_blocks.append(code_block)

    if (code_blocks is None) or (not len(code_blocks)):
        code_blocks = text
    else:
        code_blocks = code_blocks[0]

    return code_blocks


def format_code(code, tmp_dir="tmp"):
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_file_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.py")
    with open(tmp_file_path, "w", encoding="utf-8") as tmp_file:
        tmp_file.write(code)

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            subprocess.run(
                [
                    "autoflake",
                    "--in-place",
                    "--remove-all-unused-imports",
                    tmp_file_path,
                ],
                check=True,
                stdout=devnull,
                stderr=devnull,
            )
        except Exception as e:
            print(f"Error from autoflake: {e}")

        try:
            subprocess.run(
                ["python", "-m", "isort", tmp_file_path],
                check=True,
                stdout=devnull,
                stderr=devnull,
            )
        except Exception as e:
            print(f"Error from isort: {e}")

        try:
            subprocess.run(
                ["black", tmp_file_path], check=True, stdout=devnull, stderr=devnull
            )
        except Exception as e:
            _ = 1
            # print(f"Error from black: {e}")

    with open(tmp_file_path, "r", encoding="utf-8") as tmp_file:
        processed_code = tmp_file.read()
    os.remove(tmp_file_path)
    return processed_code


def strip_ansi_escape_sequences(s):
    ansi_escape_regex = re.compile(r"\x1b\[([0-9A-Za-z]+)(;[0-9]+)*m")
    return ansi_escape_regex.sub("", s)


def get_num_fails(result: subprocess.CompletedProcess[str]) -> int:
    """
    Extracts and parses the number of fails a testing run has from the pytest output.
    """
    num_fails = re.search("(\d+) failed", result.stdout)
    if num_fails is None:
        num_fails = 0
    else:
        num_fails = int(num_fails[1])
    return num_fails


def get_coverage_percent(result: subprocess.CompletedProcess[str]) -> float:
    """
    Extracts and parses the test coverage percentage from the pytest output.
    """
    coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
    if coverage_match:
        coverage_percent = coverage_match.group(1)
    else:
        coverage_percent = 0.0
    try:
        coverage_percent = float(coverage_percent)
    except ValueError as e:
        print(f"Error converting {coverage_percent} to float")
        coverage_percent = 0
    return coverage_percent


def get_test_case_fails(result: subprocess.CompletedProcess[str]) -> str:
    """
    Extracts all specific fails from the pytest output under the `short test summary info` section.
    """
    # test_cases = re.findall("(FAILED.*)$", result.stdout)
    test_cases = re.findall("(FAILED\s*test_source.py.*)", result.stdout)
    return "\n".join(test_cases)


@timeout(10)
def run_pytest(
    input_code: str,
    pytest_code: str,
    tmp_folder: str = "tmp",
    random_subdir: bool = True,
    clean_test: bool = True,
):
    current_dir = os.getcwd()
    tmp_dir_path = os.path.join(current_dir, tmp_folder)

    if random_subdir:
        random_name = "".join(
            random.choices(string.ascii_letters + string.digits, k=20)
        )
        tmp_dir_path = os.path.join(tmp_dir_path, random_name)
    os.makedirs(tmp_dir_path, exist_ok=True)

    solution_file_path = os.path.join(tmp_dir_path, "source.py")
    with open(solution_file_path, "w") as solution_file:
        solution_file.write(input_code)

    test_file_path = os.path.join(tmp_dir_path, "test_source.py")
    with open(test_file_path, "w") as test_file:
        test_file.write(pytest_code)

    try:
        result = subprocess.run(
            [
                "pytest",
                "--cov=source",
                test_file_path,
                "--cov-report",
                "term-missing",
                "-vv",
            ],
            capture_output=True,
            text=True,
            cwd=tmp_dir_path,
            timeout=5,
        )
    except Exception as e:
        print("Failed to run tests:", str(e))
        return {"coverage": 0, "failed_assertions": False, "stderr": e, "stdout": ""}

    result.stdout = strip_ansi_escape_sequences(result.stdout)

    # COVERAGE
    coverage_percent = get_coverage_percent(result)

    # NUMBER OF FAILED TESTS
    num_fails = get_num_fails(result)

    # FAIL DETAILS
    fails = get_test_case_fails(result)

    # TMP FILE CLEANUP
    if clean_test:
        shutil.rmtree(tmp_dir_path, ignore_errors=True)

    return {
        "coverage": coverage_percent,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "failed_assertions": num_fails,
        "fails": fails,
    }
