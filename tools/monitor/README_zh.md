# 训练监控日志

该监控工具用于监控远程服务器上的训练日志，检查日志中的异常情况，通过电子邮件或飞书机器人发送提醒。该工具旨在确保及时发现训练过程中的问题，包括实时监测训练干扰或减速。

# 注：

**对于电子邮件提醒:**  
   此程序需要在运行时输入密码，因此请确保在安全的环境中使用，以避免密码泄露的风险。

**对于飞书机器人提醒：**  
   此程序要求在运行时输入飞书机器人的URL，因此请确保在安全的环境中使用，以避免URL泄漏的风险。配置方法参考链接https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot，将关键字设置为 “monitor”。

训练异常监测依赖于使用各种统计方法对历史训练数据进行分析。请手动观察日志一段时间，以确保至少前10次迭代是正常的。

## 特征ge
- 监控远程日志文件以了解训练状态
- 根据日志分析结果通过电子邮件发送相应的异常提示信息，包括示例内容
- 可配置的检查间隔

## 先决条件

在运行脚本之前，请确保您能对远程主机进行免密登录【SSH】。

## 安装

   ```bash
   git clone https://github.com/FlagOpen/FlagScale.git
   cd FlagScale/tools/monitor
   pip install -r requirements.txt
   ```

## 配置

1. 对于电子邮件:

   修改提供的配置文件“config-email.yaml”示例以设置实际值：

   ```yaml
   # 接收警报的目标电子邮件地址
   target_email: example_alert@domain.com  

   # 用于发送电子邮件的 SMTP 服务器设置
   smtp_server: smtp.example.com

   # 用于发送警报的电子邮件地址
   source_email: example_sender@domain.com

   # 用于访问日志文件的远程主机 IP 地址
   remote_host: 192.0.2.1

   # SSH 登录远程主机的用户名
   remote_user: example_user

   # SSH 访问的端口号
   remote_port: 22

   # 远程主机上日志文件的路径
   remote_log_path: /path/example_log_file.log

   # 日志检查间隔（秒）
   check_interval: 1200 
   ```

2. 对于飞书机器人

   修改提供的配置文件“config feishu.yaml”示例以设置实际值：

   ```yaml
   # 用于访问日志文件的远程主机IP地址
   remote_host: 192.0.2.1

   # SSH登录的远程主机用户名
   remote_user: example_user

   # SSH 访问的端口号
   remote_port: 22

   # 远程主机上日志文件的路径
   remote_log_path: /path/example_log_file.log 

   # 日志检查间隔（秒）
   check_interval: 1200
   ```


## 用法

1. 对于邮件:

   ```bash
   python monitor.py --notice email
   ```

   然后，系统会提示您输入源电子邮件的密码。

2. 对于飞书机器人:

   ```bash
   python monitor.py --notice feishu
   ```

   然后，系统会提示您输入飞书机器人URL。

## 下一步行动

我们将添加监控视角，包括：
- 训练结束时提示
- 执行基于通信组的监控
- 监控硬件利用率异常
- 更多的用户需求 ...
