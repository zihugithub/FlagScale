import os
import re
import statistics
from typing import List, Tuple, Dict, Any

import pytest

def read_result_file(file_path: str) -> List[str]:
    """
    Read the result file and return all lines.

    Args:
        file_path (str): The path of the target file.

    Returns:
        List[str]: All lines in the file.

    Raises:
        FileNotFoundError: If the file does not exist, throw this exception.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file 'host_0_localhost.output' was not found, with the path being {file_path}")
    with open(file_path, "r", encoding='utf-8') as file:
        return file.readlines()

def parse_data_segments(lines: List[str]) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Analyze the test data and separate the data blocks that have been run multiple times from the summary data.

    Args:
        lines (List[str]): All lines read from the result file.

    Returns:
        Tuple[Dict[str, List[str]], List[str], List[str]]:
            - segments: Store data blocks for each run.
            - summary_data: Store summary data.
            - test_dataset: The running data currently being captured.
    """
    segments = {}
    summary_data = []
    test_dataset = []

    found_average_line = False  # Have you encountered the summary data separation line
    output = False
    segment_counter = 0  # Record fragment number

    for line in lines:
        # Check if there are any 'flag_gems' import failures
        assert "Failed to import 'flag_gems'" not in line, "Failed to import 'flag_gems'"

        # Determine whether it is a separated row
        if "AVERAGE PERFORMANCE OVER MULTIPLE RUNS" in line:
            found_average_line = True

        if line == "==================================================\n":
            output = True

        if line == "**************************************************\n":
            output = False
            if test_dataset:
                segment_counter += 1
                segments[f"The {segment_counter} data"] = test_dataset
                test_dataset = []

        if output:
            if not found_average_line:
                test_dataset.append(line)
            else:
                summary_data.append(line)

    return segments, summary_data, test_dataset

def extract_throughput(data_lines: List[str]) -> List[float]:
    """
    Extract throughput values from the given data rows.

    Args:
        data_lines (List[str]): A data row containing throughput information.

    Returns:
        List[float]: List of extracted throughput values.
    """
    throughput_set = []
    pattern = r'Throughput:\s*(\d+\.\d+)'
    for line in data_lines:
        match = re.search(pattern, line)
        if match:
            try:
                throughput = float(match.group(1))
                throughput_set.append(throughput)
            except ValueError:
                print(f"Warning: Unable to convert matching values to floating-point numbers: {match.group(1)}")
    return throughput_set

def calculate_statistics(data: List[float]) -> Tuple[List[float], float, float, float]:
    """
    Calculate the mean, sample variance, and sample standard deviation of the data.
    If the data length is less than 3, no pruning will be performed.

    Args:
        data (List[float]): A list of data that requires statistical calculations.

    Returns:
        Tuple[List[float], float, float, float]:
            - trimmed_data: Trimmed data list.
            - mean: Mean value.
            - variance: Sample variance.
            - std_dev: Sample standard deviation.
    """
    if len(data) < 3:
        trimmed_data = data
    else:
        # Calculate statistical measures after removing the maximum and minimum values
        sorted_data = sorted(data)
        trimmed_data = sorted_data[1:-1]

    mean = statistics.mean(trimmed_data)
    variance = statistics.variance(trimmed_data) if len(trimmed_data) > 1 else 1.3838
    std_dev = statistics.stdev(trimmed_data) if len(trimmed_data) > 1 else 1.1763

    return trimmed_data, mean, variance, std_dev

def display_results(throughput_set: List[float], trimmed_data: List[float],
                    mean: float, variance: float, std_dev: float, summary_data: List[str]):
    """
    Print throughput collection, trimmed data, and statistical results.

    Args:
        throughput_set (List[float]): Raw throughput data.
        trimmed_data (List[float]): Trimmed throughput data.
        mean (float): Mean value.
        variance (float): Sample variance.
        std_dev (float): Sample standard deviation.
        summary_data (List[str]): Summarize data.
    """
    print("Throughput set:", throughput_set)
    print("Trimmed data:", trimmed_data)
    print(f"mean: {mean:.4f}")
    print(f"Sample variance: {variance:.4f}")
    print(f"Sample standard deviation: {std_dev:.4f}")
    print("\nSummary data:")
    for line in summary_data:
        print(line, end='')

def is_within_tolerance(result: List[float], gold_values: List[float], std_dev: float) -> bool:
    """
    Check if the results are within the allowable tolerance range.

    Args:
        result (List[float]): Test result throughput.
        gold_values (List[float]): Benchmark throughput value.
        std_dev (float): Standard deviation.

    Returns:
        bool: If the result is within the tolerance range, it is True; Otherwise, it is False.

    Raises:
        ValueError: If gold_values is empty, throw this exception.
    """
    if not gold_values:
        raise ValueError("The gold_values list cannot be empty.")

    baseline = gold_values[0]
    lower_bound = baseline - std_dev
    upper_bound = baseline + std_dev

    return all(lower_bound <= val <= upper_bound for val in result)

@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_throughput_equal(test_path: str, test_type: str, test_task: str, test_case: str):
    """
    Test whether the throughput meets expectations.

    Args:
        test_path (str): Test path.
        test_type (str): Test type.
        test_task (str): Test task.
        test_case (str): Test cases.
    """
    print("test_path")
    # Construct the test_result_path and gold_value_path using the provided fixtures
    test_result_path = os.path.join(test_path, test_type, test_task, "results_test", test_case)
    result_path = os.path.join(test_result_path, "inference_logs/host_0_localhost.output")
    gold_value_path = os.path.join(test_path, test_type, test_task, "results_gold", test_case)

    print("result_path:", result_path)
    print("gold_value_path:", gold_value_path)

    try:
        result_lines = read_result_file(result_path)
        gold_value_lines = read_result_file(gold_value_path)
    except FileNotFoundError as e:
        print(e)
        pytest.fail("Test failed due to missing necessary files.")

    try:
        segments, summary_data, test_dataset = parse_data_segments(result_lines)
    except AssertionError as ae:
        print(ae)
        pytest.fail("An assertion error occurred while parsing the data segment.")

    print("\nResult checking")
    all_throughput = []

    if segments:
        # Multiple throughput tests run
        for seg_name, seg_lines in segments.items():
            throughput = extract_throughput(seg_lines)
            all_throughput.extend(throughput)
        print("Result summary: ", segments)
    elif test_dataset and not summary_data:
        # Single run throughput test
        summary_data = test_dataset
        seg_lines = test_dataset
        throughput = extract_throughput(seg_lines)
        all_throughput.extend(throughput)
        print("Result summary: ", seg_lines)
    else:
        print("Failed to identify valid data segments.")

    print("Gold Result: ", gold_value_lines)

    if not all_throughput:
        print("Failed to extract any throughput data.")
        pytest.fail("Failed to extract throughput data.")

    trimmed_data, mean, variance, std_dev = calculate_statistics(all_throughput)
    
    gold_throughput = extract_throughput(gold_value_lines)
    result_throughput = extract_throughput(summary_data)

    try:
        assert is_within_tolerance(result_throughput, gold_throughput, std_dev), f"{result_throughput} exceeds the allowed range (± {std_dev}) for gold_throughput ({gold_throughput})."
        display_results(all_throughput, trimmed_data, mean, variance, std_dev, summary_data)
        print(f"{result_throughput} is within the allowed range of gold_throughput ({gold_throughput}) (± {std_dev}).")
    except ValueError as ve:
        print(ve)
    except AssertionError as ae:
        print(ae)
        pytest.fail("Throughput test failed.")
