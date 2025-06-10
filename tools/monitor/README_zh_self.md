**此文件仅作为对monitor.py的解读**
```py
parser = argparse.ArgumentParser(description="Training Monitor") # 创建参数解析器
# 通过 add_argument() 方法定义具体的命令行参数
parser.add_argument(
    name,    # 参数名称（如 '--input' 或 'input'）
             # 必选参数：直接写名称（如 input）
             # 可选参数：以 -- 或 - 开头（如 --input 或 -i）
    kwargs   # 可选参数
)
# 解析命令行参数
args = parser.parse_args()

ssh = paramiko.SSHClient()  # 通过 SSH 协议与远程服务器建立安全连接
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 自动添加未知主机密钥
ssh.connect(hostname=remote_host, username=remote_user, port=remote_port) # 连接远程服务器

sftp = ssh.open_sftp() # 创建 sftp 会话，通过 SFTP 协议在本地和远程服务器之间传输文件

with sftp.open(remote_log_path, "r") as remote_file:
            contents = remote_file.readlines()
sftp.close()
ssh.close()

match = re.search(pattern, line) # 扫描整个字符串并返回第一个成功的匹配
parsed_data = match.groupdict()  # 从命名捕获组中提取匹配的内容，并以字典的形式返回

recent_logs = logs[-10:]  # 提取最后 10 个元素
last_log = logs[-1]  # 提取最后一个元素
#logs[-5:]：提取最后 5 条日志。
#logs[:10]：提取前 10 条日志。
#logs[5:15]：提取第 6 到第 15 条日志（索引从 0 开始
last_timestamp = datetime.strptime(
   last_log["timestamp"], "%Y-%m-%d %H:%M:%S"
) # 字符串转时间

time_delta_ms = (now - last_timestamp).total_seconds() * 1000  # 以毫秒为单位计算时间差
```

```py
pattern = (
    # [2023-10-05 14:30:45]
    # r 表示原始字符串，避免转义字符的干扰
    # \[+  匹配左方括号，+ 表示前一个字符出现一次或多次
    # (?P<timestamp>...) 命名捕获组，将匹配的内容命名为 timestamp
    # \d{4} 匹配4个数字
    # - 匹配连字符
    # \s* 匹配零个或多个空白字符
    # () 有特殊含义为捕获组，需要使用 \ 转义
    r"\[+(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]+\s*"
    r"iteration\s+(?P<iteration>\d+\s*/\s*\d+)\s+\|\s*"
    r"consumed samples:\s+(?P<consumed_samples>\d+)\s+\|\s*"
    r"elapsed time per iteration \(ms\):\s+(?P<elapsed_time_per_iteration_ms>\d+\.\d+)\s+\|\s*"
    r"throughput per GPU \(TFLOP/s/GPU\):\s+(?P<throughput_per_GPU_TFLOPs_per_GPU>\d+\.\d+)\s+\|\s*"
    r"learning rate:\s+(?P<learning_rate>\d+\.?\d*E?-?\d*)\s+\|\s*"
    r"global batch size:\s+(?P<global_batch_size>\d+)\s+\|\s*"
    r"lm loss:\s+(?P<lm_loss>\d+\.\d+E?[-\+]?\d*)\s+\|\s*"
    r"load_balancing_loss:\s+(?P<load_balancing_loss>\d+\.\d+E?[-\+]?\d*)\s+\|\s*"
    r"loss scale:\s+(?P<loss_scale>\d+\.\d+)\s+\|\s*"
    r"grad norm:\s+(?P<grad_norm>\d+\.\d+)\s+\|\s*"
    r"num zeros:\s+(?P<num_zeros>[\d.]+)\s+\|\s*"
    r"params norm:\s+(?P<params_norm>\d+\.\d+)\s+\|\s*"
    r"number of skipped iterations:\s+(?P<number_of_skipped_iterations>\d+)\s+\|\s*"
    r"number of nan iterations:\s+(?P<number_of_nan_iterations>\d+)\s*"
)

lines_to_keep = []
for line in log_lines:
    match = re.search(pattern, line)
    if match:
        parsed_data = match.groupdict()
        if parsed_data["iteration"].startswith("1/"):  
            lines_to_keep = []   # 如果 iteration 字段以 "1/" 开头，表示新的阶段开始，清空 lines_to_keep 列表
        lines_to_keep.append(parsed_data)
```

[default7]: [2025-02-28 16:25:34] iteration        1/  119209 | consumed samples:         2048 | elapsed time per iteration (ms): 46298.6 | throughput per GPU (TFLOP/s/GPU): 219.0 | learning rate: 3.000000E-06 | global batch size:  2048 | lm loss: 1.233673E+01 | load_balancing_loss: 9.937914E-01 | loss scale: 1.0 | grad norm: 5.334 | num zeros: 111393856.0 | params norm: 2120.030 | number of skipped iterations:   0 | number of nan iterations:   0 |
用于监控训练过程是否正常进行，以及模型是否在收敛
基本信息：
- 任务名称：default7
- 总迭代次数: 119209
- 全局批量大小: 2048 （每次迭代处理2048个样本）
- 学习率: 随着迭代逐步增加（从3e-6到1.2e-5）
- 损失缩放: 1.0（用于防止数值下溢或溢出）
- 跳过迭代次数: 0（没有跳过任何迭代）
- NaN迭代次数: 0（没有出现NaN值）
性能指标
- 每次迭代耗时: 46298.6毫秒
- GPU吞吐量（TFLOP/s/GPU）： 219.0
- 梯度范数（grad norm）： 5.334
- 参数范数（params norm）: 2120.030
损失函数
- lm loss（语言模型损失）：12.33673
- load_balancing_loss（负载均衡损失）：0.9937914
稀疏性指标
- num zeros（零值数量）: 111,393,856.0
分析与观察
1. 性能提升：
  - 每次迭代的耗时逐渐减少（从46.3秒到29.7秒），表明训练过程逐渐稳定。
  - GPU吞吐量逐步提高（从219.0到341.9 TFLOP/s/GPU），说明计算效率在提升。
2. 损失下降：
  - lm loss 和 load_balancing_loss 均呈现下降趋势，表明模型正在收敛。
3. 梯度范数减小：
  - grad norm 从5.334逐步下降到4.262，说明梯度更新趋于稳定。
4. 稀疏性变化：
  - num zeros 波动较大，但整体呈下降趋势，可能与模型参数的稀疏性有关。
5. 学习率调整：
  - 学习率从3e-6逐步增加到1.2e-5，可能是为了加速收敛。