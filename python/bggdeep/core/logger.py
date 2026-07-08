"""
BggDeepLearning 日志模块

文件位置：
python/bggdeep/core/logger.py

作用：
1. 创建统一日志系统
2. 同时把日志输出到终端和日志文件
3. 日志文件默认保存到 outputs/logs/bggdeep.log
4. 后续所有模块都可以通过 get_logger() 获取同一个日志对象

为什么要用日志？

新手阶段我们常用 print()。
但是大型项目中，print() 有几个问题：
1. 不方便保存历史记录
2. 不方便区分 INFO / WARNING / ERROR
3. 不方便定位错误来自哪个文件
4. 后期前后端、模型训练、报告生成都会产生大量信息

所以从现在开始，我们逐步使用 logging 日志系统。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


LOGGER_NAME = "bggdeep"


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件位置是：
    BggDeepLearning/python/bggdeep/core/logger.py

    所以：
    logger.py 的上一级是 core
    再上一级是 bggdeep
    再上一级是 python
    再上一级就是 BggDeepLearning 项目根目录
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    return project_root


def setup_python_path(project_root: Optional[Path] = None) -> None:
    """
    把 BggDeepLearning/python 加入 Python 搜索路径。

    这样我们就可以在不同文件中写：

    from bggdeep.core.config import load_yaml_config
    """
    if project_root is None:
        project_root = get_project_root()

    python_dir = project_root / "python"

    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    从配置字典中读取指定层级的值。

    示例：
    key_path = "logging.level"

    等价于：
    config["logging"]["level"]
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


def build_log_formatter() -> logging.Formatter:
    """
    创建日志格式。

    日志格式示例：
    2026-07-07 13:30:00 | INFO | bggdeep | main.py:120 | 项目启动成功
    """
    log_format = (
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(filename)s:%(lineno)d | "
        "%(message)s"
    )

    date_format = "%Y-%m-%d %H:%M:%S"

    return logging.Formatter(fmt=log_format, datefmt=date_format)


def setup_logger(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    初始化日志系统。

    参数：
    config:
        项目配置字典。如果传入，就从配置中读取日志级别和日志文件路径。
        如果不传入，就使用默认值。

    返回：
    logging.Logger 对象。

    注意：
    为了避免重复添加 handler，这里会先清空旧 handler。
    """
    project_root = get_project_root()

    log_level_text = "INFO"
    log_file_text = "outputs/logs/bggdeep.log"

    if config is not None:
        log_level_text = get_config_value(config, "logging.level", log_level_text)
        log_file_text = get_config_value(config, "logging.log_file", log_file_text)

    log_level = getattr(logging, str(log_level_text).upper(), logging.INFO)

    log_file = project_root / log_file_text
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)

    # 防止重复运行时重复输出日志
    logger.handlers.clear()

    formatter = build_log_formatter()

    # 1. 输出到终端
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # 2. 输出到日志文件
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 防止日志向 root logger 继续传播，避免重复显示
    logger.propagate = False

    logger.info("日志系统初始化成功")
    logger.info("日志文件路径：%s", log_file)

    return logger


def get_logger() -> logging.Logger:
    """
    获取项目统一 logger。

    如果 logger 还没有初始化，则使用默认配置初始化。
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        logger = setup_logger()

    return logger


def test_logger() -> None:
    """
    测试日志模块是否可用。

    运行方式：
    python python\\bggdeep\\core\\logger.py
    """
    setup_python_path()

    try:
        from bggdeep.core.config import load_yaml_config

        config = load_yaml_config("app.yaml")
    except Exception as exc:
        print("配置文件读取失败，将使用默认日志配置。")
        print(f"错误信息：{exc}")
        config = None

    logger = setup_logger(config)

    logger.debug("这是一条 DEBUG 日志，一般用于调试细节。")
    logger.info("这是一条 INFO 日志，表示普通运行信息。")
    logger.warning("这是一条 WARNING 日志，表示需要注意但不一定中断。")
    logger.error("这是一条 ERROR 日志，表示出现错误。")

    print("=" * 60)
    print("logger.py 测试运行完成")
    print("请检查 outputs\\logs\\bggdeep.log 是否生成")
    print("=" * 60)


if __name__ == "__main__":
    test_logger()