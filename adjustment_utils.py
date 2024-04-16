import re
import ast, astor

from utils import *

def convert_to_ast_node(value_str):
    try:
        value = ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        try:
            value = eval(value_str)
        except:
            value = value_str

    return ast.parse(repr(value)).body[0].value

def replace_assert_value(node, actual_str, expected_str):
    modified = False
    for child in ast.walk(node):
        if isinstance(child, ast.Compare) and len(child.comparators) == 1:
            actual_value_str = astor.to_source(child.comparators[0]).strip()

            try:
                actual_value_eval = eval(actual_value_str)
                actual_str_eval = eval(actual_str)
                eval_match = actual_value_eval == actual_str_eval
            except (ValueError, SyntaxError) as e:
                # print(e)
                eval_match = False

            if actual_value_str == actual_str or actual_value_str.strip("()") == actual_str or eval_match:
                child.comparators[0] = convert_to_ast_node(expected_str)
                modified = True
                break    
    return modified




def replace_variable_with_literal(node, variable_name, new_value):
    """
    Replaces references to a given variable in assert statements with a literal value.
    """
    modified = False
    for child in ast.walk(node):
        if isinstance(child, ast.Compare) and len(child.comparators) == 1:
            if isinstance(child.comparators[0], ast.Name) and child.comparators[0].id == variable_name:
                child.comparators[0] = convert_to_ast_node(new_value)
                modified = True
    return modified

def adjust_pytest(pytest, output):
    """
    Adjusts pytest assertions based on the output.
    """
    pytest_lines = pytest.split('\n')
    failure_report_strings = re.split("_{3,} test.* _{3,}", output["stdout"])[1:]
    failure_report_strings[-1] = re.split("---------- coverage", failure_report_strings[-1])[0]


    for report in failure_report_strings:
        line_number = int(re.findall("test_source.py:(\d+):", report)[0]) - 1
        line_number = min(line_number, len(pytest_lines)-1)

        leading_whitespace = re.match("^(\s*)", pytest_lines[line_number])
        if not leading_whitespace:
            leading_whitespace = ""
        else:
            leading_whitespace = leading_whitespace.group(0)


        if "pytest.approx" in pytest_lines[line_number]:
            pytest_lines[line_number] = re.sub("pytest\.approx\(([^,)]+),?[^)]*\)", lambda m : f"{ m.group(1) }", pytest_lines[line_number])
            continue
        

        error_type = re.findall("source\.py.*\: ((.*)Error)", report)
        if error_type:
            if isinstance(error_type, list):
                error_type = error_type[-1]

            if (error_type[1].lower() != "assertion") and (error_type[1].lower() != "synthax"):
                
                if not any(["import pytest" in line for line in pytest_lines]):
                    pytest_lines =  ["import pytest"] + pytest_lines
                    line_number += 1
                
                pytest_lines[line_number] = pytest_lines[line_number].strip()
                pytest_lines[line_number] = leading_whitespace + f"with pytest.raises({error_type[0]}):\n" +\
                    leading_whitespace + " "* 4 + pytest_lines[line_number]
                continue


        if " and " in pytest_lines[line_number]:
            conditions = re.findall("(.*assert)\s*(.*)", pytest_lines[line_number])
            if conditions:
                if isinstance(conditions, list):
                    conditions = conditions[0]
                prefix, conditions = conditions
                conditions = conditions.split("and")
                conditions = [f"{prefix} {condition}" for condition in conditions]
                conditions = "\n".join(conditions)

                pytest_lines[line_number] = conditions
                continue


        if "assert" not in pytest_lines[line_number].lower():
            line_number+=1


        if "assert false" in report.lower():
            pytest_lines[line_number] = re.sub("(.*assert)(.*)", lambda m : f"{m.group(1)} not {m.group(2)}", pytest_lines[line_number])
            continue
        


        pytest_lines[line_number] = pytest_lines[line_number].strip() 
        match = re.search(r"assert\s*(.+)\s*==\s*(.+)", pytest_lines[line_number])
        if match:
            left, right = match.groups()

            if right.strip().isidentifier() and right.strip() not in ["True", "False"]:
                actual_value = re.findall("assert(?:\s+)(.*) ==", report)[-1]

                if actual_value:
                    new_value = actual_value.strip()

                    tree = ast.parse(pytest_lines[line_number], mode='exec')
                    modified = replace_variable_with_literal(tree, right, new_value)
                    if modified:
                        modified_line = astor.to_source(tree).strip()
                        pytest_lines[line_number] = leading_whitespace + modified_line
            else:
                expected, actual = re.findall("assert\s*(.*) == (.*)", report)[-1]
                tree = ast.parse(pytest_lines[line_number], mode='exec')
                modified = replace_assert_value(tree, actual.strip(), expected.strip())

                if modified:
                    modified_line = astor.to_source(tree).strip()
                    pytest_lines[line_number] = modified_line
            pytest_lines[line_number] = leading_whitespace + pytest_lines[line_number]
            
    return '\n'.join(pytest_lines)


def strip_ansi_escape_sequences(s):
    ansi_escape_regex = re.compile(r'\x1b\[([0-9A-Za-z]+)(;[0-9]+)*m')
    return ansi_escape_regex.sub('', s)


def modify_pytest_code(code, pytest):
    """
    Modifies pytest code based on test execution results.
    """
    prev_out = {"stdout":  None}
    #remove comments
    pytest = re.sub("#.*\n", "\n", pytest)
    #remove multiple spaces at the end
    pytest = re.sub("\s*(\n)+", "\n", pytest, count=500)
    #remove duplicated objects
    code_classes = get_object_names_from_code(code)
    pytest_classes = get_object_names_from_code(code)
    duplicated_objects = code_classes.union(pytest_classes)
    for duplicated_obj in duplicated_objects:
        pytest = delete_object_from_code(duplicated_obj, pytest)


    for _ in range(35):
        out = run_pytest(code, pytest)
        if out["stdout"] == prev_out["stdout"]:
            return pytest
        else:
            prev_out["stdout"] = out["stdout"]
        if "NameError" in out["stdout"] and "from source import *" not in pytest:
            pytest = "from source import *\n" + pytest
            continue

        if not out["failed_assertions"]:
            return pytest
        
        pytest = adjust_pytest(pytest, out)

    if out["failed_assertions"] or not pytest["coverage"]:
        return None 
        
    return pytest