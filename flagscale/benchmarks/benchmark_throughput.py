import os
import sys
import time
import yaml
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import psutil
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config

@dataclass
class PerformanceMetrics:
    """Performance indicator data category"""
    throughput: float          # throughput (tokens/second)
    latency: float             # delay (seconds)
    total_tokens: int          # Total number of tokens processed
    input_tokens: int          # Enter the number of tokens
    output_tokens: int         # Output the number of tokens
    gpu_memory_usage: float    # GPU memory usage rate (GB)
    cpu_utilization: float     # CPU utilization (%)

class ThroughputCalculator:
    """Throughput Calculator"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        """Start timer"""
        self.start_time = time.time()
        print("Starting inference...")

    def stop_timer(self):
        """Stop timer"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"Inference completed in {duration:.2f} seconds")
        return duration

    def count_tokens(self, inputs: List[Dict], outputs: List[Any], tokenizer) -> Dict[str, int]:
        """Count the number of tokens"""
        input_tokens = 0
        output_tokens = 0

        for i, input_data in enumerate(inputs):
            prompt = input_data["prompt"]
            # Using a tokenizer to count input tokens
            encoded = tokenizer.encode(prompt)
            input_tokens += len(encoded)

            # Statistics output token
            if i < len(outputs):
                output_tokens += len(outputs[i].outputs[0].token_ids)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

    def calculate_metrics(self, token_counts: Dict[str, int], duration: float) -> PerformanceMetrics:
        """Calculate performance indicators"""
        total_tokens = token_counts["total_tokens"]
        throughput = total_tokens / duration if duration > 0 else 0

        # Get GPU memory usage (in GB)
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB

        # Obtain CPU utilization rate
        cpu_usage = psutil.cpu_percent(interval=duration)

        return PerformanceMetrics(
            throughput=throughput,
            latency=duration,
            total_tokens=total_tokens,
            input_tokens=token_counts["input_tokens"],
            output_tokens=token_counts["output_tokens"],
            gpu_memory_usage=gpu_memory,
            cpu_utilization=cpu_usage
        )

    def print_metrics(self, metrics: PerformanceMetrics):
        """Printing performance indicators"""
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Throughput: {metrics.throughput:.2f} tokens/second")
        print(f"Total Latency: {metrics.latency:.2f} seconds")
        print(f"Total Tokens Processed: {metrics.total_tokens}")
        print(f"Input Tokens: {metrics.input_tokens}")
        print(f"Output Tokens: {metrics.output_tokens}")
        print(f"GPU Memory Usage: {metrics.gpu_memory_usage:.2f} GB")
        print(f"Average CPU utilization rate: {metrics.cpu_utilization:.1f}%")
        print("="*50)

def enhanced_inference(cfg) -> PerformanceMetrics:
    """Enhanced inference function, including throughput calculation"""
    
    # Step 1: Analyze the configuration
    try:
        generate_cfg = cfg.get("generate", {})
        prompts_file = generate_cfg.get("prompts_file")

        if prompts_file:
            script_path = os.path.abspath(__file__)
            script_dir = os.path.dirname(script_path)
            prompts_file = f"{script_dir}/{prompts_file}"
            print(f"prompts_file: {prompts_file}")
            if not os.path.exists(prompts_file):
                raise FileNotFoundError(f"Prompts file '{prompts_file}' does not exist.")
            with open(prompts_file, 'r') as pf:
                prompts = yaml.safe_load(pf)
            if not isinstance(prompts, list):
                raise ValueError(f"Prompts file '{prompts_file}' should contain a list of strings.")
            print(f"Loaded {len(prompts)} prompts from '{prompts_file}'.")
        else:
            prompts = generate_cfg.get("prompts", [])
            if not prompts:
                raise ValueError("No prompts found in config or external file.")

    except Exception as e:
        print(f"Error reading prompts from config: {e}")
        sys.exit(1)

    # Step 2: Initialize LLM engine
    llm_cfg = cfg.get("llm", {})
    try:
        llm = LLM(**llm_cfg)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    # Set up a tokenizer
    tokenizer_path = llm_cfg.get("tokenizer", None)
    if tokenizer_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            llm.set_tokenizer(tokenizer)
        except Exception as e:
            print(f"Error setting tokenizer: {e}")
            sys.exit(1)
    else:
        tokenizer = None
        print("Tokenizer path not provided. Using default tokenizer.")

    # Step 3: Initialize sampling parameters
    sampling_cfg = cfg.generate.get("sampling", {})
    if sampling_cfg.get("logits_processors"):
        print("logits_processors is not supported yet.")
        sys.exit(1)
    try:
        sampling_params = SamplingParams(**sampling_cfg)
        print(f"=> Sampling Parameters: {sampling_params}")
    except Exception as e:
        print(f"Error setting sampling parameters: {e}")
        sys.exit(1)

    # Step 4: Build Input
    inputs = [{"prompt": prompt} for prompt in prompts]
    print(f"=> Inputs prepared: {len(inputs)} prompts")

    # Step 5: Perform inference and calculate throughput
    calculator = ThroughputCalculator()
    calculator.start_timer()

    try:
        outputs = llm.generate(inputs, sampling_params)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

    duration = calculator.stop_timer()
    token_counts = calculator.count_tokens(inputs, outputs, tokenizer)
    metrics = calculator.calculate_metrics(token_counts, duration)

    # Step 6: Output results and performance metrics
    calculator.print_metrics(metrics)

    # Display generated results
    for i, output in enumerate(outputs):
        print(f"\n{'*' * 50}")
        print(f"{output.prompt=}")
        print(f"{output.outputs[0].text=}")
        print(f"{output.outputs[0].token_ids=}")
    print("#" * 50)

    return metrics

def benchmark_multiple_runs(cfg, num_runs: int = 3) -> PerformanceMetrics:
    """Run benchmark tests multiple times and take the average value"""
    all_metrics = []

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        metrics = enhanced_inference(cfg)
        all_metrics.append(metrics)

        if run < num_runs - 1:
            print("Sleeping for 5 seconds to avoid overheating...")
            time.sleep(5)

    # Calculate the average indicator
    if all_metrics:
        avg_metrics = PerformanceMetrics(
            throughput=sum(m.throughput for m in all_metrics) / num_runs,
            latency=sum(m.latency for m in all_metrics) / num_runs,
            total_tokens=all_metrics[-1].total_tokens,  # The last total number of tokens
            input_tokens=all_metrics[-1].input_tokens,
            output_tokens=all_metrics[-1].output_tokens,
            gpu_memory_usage=all_metrics[-1].gpu_memory_usage,
            cpu_utilization=all_metrics[-1].cpu_utilization
        )

        print("\nAVERAGE PERFORMANCE OVER MULTIPLE RUNS")
        print("=" * 40)
        calculator = ThroughputCalculator()
        calculator.print_metrics(avg_metrics)
    else:
        print("No metrics were collected.")
        avg_metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)

    return avg_metrics

if __name__ == "__main__":
    try:
        cfg = parse_config()
    except Exception as e:
        print(f"Error parsing configuration: {e}")
        sys.exit(1)

    print(f"cfg: {cfg}")

    # Single run or multiple throughput tests
    use_benchmark = cfg.get("use_benchmark", False)

    if use_benchmark:
        num_runs = cfg.get("num_runs", 3)
        benchmark_multiple_runs(cfg, num_runs)
    else:
        enhanced_inference(cfg)
