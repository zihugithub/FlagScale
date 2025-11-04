import json
import os

import numpy as np
import pytest
import requests

from omegaconf import DictConfig, OmegaConf

@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_inference_equal(test_path, test_type, test_task, test_case):
    # Construct the test_result_path using the provided fixtures
    test_result_path = os.path.join(test_path, test_type, test_task, "results_test", test_case)
    result_path = os.path.join(test_result_path, "inference_logs/host_0_localhost.output")

    print("result_path:", result_path)

    assert os.path.exists(result_path), f"Failed to find 'host_0_localhost.output' at {result_path}"

    with open(result_path, "r") as file:
        lines = file.readlines()

    result_lines = []
    output = False
    for line in lines:
        assert "Failed to import 'flag_gems'" not in line, "Failed to import 'flag_gems''"
        if line == "============================================================\n":
            output = True
        if line == "************************************************************\n":
            output = False
        if output == True:
            result_lines.append(line)

    gold_value_path = os.path.join(test_path, test_type, test_task, "results_gold", test_case)
    assert os.path.exists(gold_value_path), f"Failed to find gold result at {gold_value_path}"

    with open(gold_value_path, "r") as file:
        gold_value_lines = file.readlines()

    # Remove the blank line at the end.
    if gold_value_lines:
        last_non_empty = len(gold_value_lines) - 1
        while last_non_empty >= 0 and not gold_value_lines[last_non_empty].strip():
            last_non_empty -= 1
        if last_non_empty >= 0:
            gold_value_lines = gold_value_lines[: last_non_empty + 1]
        else:
            gold_value_lines = []

    print("\nResult checking")
    print("Result: ", result_lines)
    print("Gold Result: ", gold_value_lines)

    print("len(result_lines), (gold_value_lines): ", len(result_lines), len(gold_value_lines))
    assert len(result_lines) == len(gold_value_lines)

    for result_line, gold_value_line in zip(result_lines, gold_value_lines):
        print(result_line, gold_value_line)
        assert result_line.rstrip('\n') == gold_value_line.rstrip('\n')
