# hydra 框架简介
**hydra** 作为 Python 配置管理框架（机器学习/工程领域），在 AI 和软件开发中，Hydra 基于 OmegaConf 库的配置管理系统，用于简化复杂项目的多层级参数管理
- 由 Facebook AI 开发
- 通过组合动态创建分层配置
- 支持通过配置文件或命令行覆盖配置

# hydra 的使用
**安装**
```sh
pip install hydra-core --upgrade
pip install omegaconf
```
创建config.yml
```yaml
database:
  driver: mysql
  user: root
  password: secret

```
**编写python案例** script.py

使用装饰器 `@hydra.main()` 解析参数并注入到函数中
```sh
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    print(f"Database driver: {cfg.database.driver}")
    print(f"User: {cfg.database.user}")

if __name__ == "__main__":
    main()
```
运行脚本并通过命令行覆盖配置

```sh
python script.py database.driver=postgresql database.user=admin
```

**hydra 结构化配置**

Hydra 允许将配置文件拆分成多个部分，增强可读性和可维护性。例如，我们可以创建一个 config 文件夹，并拆分配置

案例1
```
config/   
│── config1.yaml   
│── model.yaml   
│── data.yaml
```
- config1.yaml
```yaml
defaults:
  - model: resnet
  - data: dataset1
```
- model.yaml
```yaml
name: "resnet50"
learning_rate: 0.01
```
- data.yaml
```yaml
dataset_name: "CIFAR-10"
batch_size: 64
```

- traun.py
```py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    print(f"Model: {cfg.model.name}, Learning Rate: {cfg.model.learning_rate}")
    print(f"Dataset: {cfg.data.dataset_name}, Batch Size: {cfg.data.batch_size}")

if __name__ == "__main__":
    train()
```

案例2

```
config/   
│── config2.yaml   
│── inference  
│──── data.yaml
```

- config2.yaml
```yaml
defaults:
  - _self_
  - inference: data
```
- data.yaml
```yaml
dataset_name: "CIFAR-10"
batch_size: 64
```