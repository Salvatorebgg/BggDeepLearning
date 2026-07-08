"""
BggDeepLearning 配置读取模块

文件位置：
python/bggdeep/core/config.py

作用：
1. 找到项目根目录
2. 找到 configs/app.yaml
3. 读取 YAML 配置
4. 为后续数据、模型、日志、报告等模块提供统一配置
"""

from pathlib import Path
from typing import Any, Dict


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件位置是：
    BggDeepLearning/python/bggdeep/core/config.py

    所以：
    config.py 的上一级是 core
    再上一级是 bggdeep
    再上一级是 python
    再上一级就是 BggDeepLearning 项目根目录
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    return project_root


def get_config_path(config_name: str = "app.yaml") -> Path:
    """
    获取配置文件路径。

    默认读取：
    configs/app.yaml
    """
    project_root = get_project_root()
    config_path = project_root / "configs" / config_name
    return config_path


def load_yaml_config(config_name: str = "app.yaml") -> Dict[str, Any]:
    """
    读取 YAML 配置文件。

    参数：
    config_name: 配置文件名，默认是 app.yaml

    返回：
    一个 Python 字典 dict
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "未安装 PyYAML，请先在 VS Code 终端执行：python -m pip install pyyaml"
        ) from exc

    config_path = get_config_path(config_name)

    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在：{config_path}\n"
            "请检查 configs/app.yaml 是否已经创建。"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"配置文件为空：{config_path}\n"
            "请检查 app.yaml 是否已经写入内容。"
        )

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    按层级读取配置值。

    例如：
    key_path = "project.name"

    会读取：
    config["project"]["name"]

    如果找不到，就返回 default。
    """
    keys = key_path.split(".")
    value: Any = config

    for key in keys:
        if not isinstance(value, dict):
            return default

        if key not in value:
            return default

        value = value[key]

    return value


def print_config_summary() -> None:
    """
    打印配置摘要。

    这个函数主要用于测试：
    1. 配置文件是否能找到
    2. YAML 是否能读取
    3. 关键字段是否正常
    """
    config = load_yaml_config()

    project_name = get_config_value(config, "project.name", "未知项目")
    project_version = get_config_value(config, "project.version", "未知版本")
    project_stage = get_config_value(config, "project.stage", "未知阶段")
    main_topic = get_config_value(config, "clinical_task.main_topic", "未知主题")
    log_dir = get_config_value(config, "paths.log_dir", "outputs/logs")

    print("=" * 60)
    print("BggDeepLearning 配置文件读取成功")
    print(f"项目名称：{project_name}")
    print(f"项目版本：{project_version}")
    print(f"当前阶段：{project_stage}")
    print(f"临床主题：{main_topic}")
    print(f"日志目录：{log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()