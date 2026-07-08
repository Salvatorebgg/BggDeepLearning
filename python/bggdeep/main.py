"""
BggDeepLearning 项目主入口文件

文件位置：
python/bggdeep/main.py

当前功能：
1. 获取项目根目录
2. 把 python 目录加入模块搜索路径
3. 读取 configs/app.yaml 配置文件
4. 初始化正式日志系统
5. 初始化项目路径系统
6. 自动确认项目常用文件夹存在
7. 打印项目启动信息
8. 写入启动日志

运行方式：
在项目根目录执行：
python python\\bggdeep\\main.py
"""

from pathlib import Path
from datetime import datetime
import sys
from typing import Any, Dict


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件位置是：
    BggDeepLearning/python/bggdeep/main.py
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    return project_root


def setup_python_path(project_root: Path) -> None:
    """
    设置 Python 模块搜索路径。

    把：
    BggDeepLearning/python

    加入 sys.path。
    """
    python_dir = project_root / "python"

    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def load_project_config() -> Dict[str, Any]:
    """
    加载项目配置文件。
    """
    from bggdeep.core.config import load_yaml_config

    config = load_yaml_config("app.yaml")
    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    从配置字典中读取指定层级的值。
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


def write_startup_log(
    project_root: Path,
    config: Dict[str, Any],
    log_dir: Path,
) -> None:
    """
    写入一个简单启动检查文件。

    注意：
    这个文件是 startup_check.txt。
    更完整的正式日志写入 bggdeep.log。
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "startup_check.txt"

    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    project_name = get_config_value(config, "project.name", "BggDeepLearning")
    project_version = get_config_value(config, "project.version", "0.1.0")
    project_stage = get_config_value(config, "project.stage", "unknown")
    main_topic = get_config_value(config, "clinical_task.main_topic", "unknown")

    log_text = (
        f"启动时间：{now_time}\n"
        f"项目名称：{project_name}\n"
        f"项目版本：{project_version}\n"
        f"当前阶段：{project_stage}\n"
        f"临床主题：{main_topic}\n"
        f"项目根目录：{project_root}\n"
        f"配置文件：configs/app.yaml\n"
        f"正式日志文件：outputs/logs/bggdeep.log\n"
        f"路径管理模块：python/bggdeep/core/paths.py\n"
        f"启动状态：成功\n"
    )

    log_file.write_text(log_text, encoding="utf-8")


def print_startup_summary(
    project_root: Path,
    config: Dict[str, Any],
    paths_summary: Dict[str, Path],
) -> None:
    """
    在终端打印启动摘要。
    """
    project_name = get_config_value(config, "project.name", "BggDeepLearning")
    project_full_name = get_config_value(config, "project.full_name", "")
    project_version = get_config_value(config, "project.version", "0.1.0")
    project_stage = get_config_value(config, "project.stage", "unknown")
    main_topic = get_config_value(config, "clinical_task.main_topic", "unknown")
    outcome_name = get_config_value(config, "clinical_task.outcome_name", "poor_outcome")
    log_file = get_config_value(config, "logging.log_file", "outputs/logs/bggdeep.log")

    print("=" * 70)
    print("BggDeepLearning Python 主程序启动成功")
    print("-" * 70)
    print(f"项目名称：{project_name}")
    print(f"项目全称：{project_full_name}")
    print(f"项目版本：{project_version}")
    print(f"当前阶段：{project_stage}")
    print(f"项目根目录：{project_root}")
    print("-" * 70)
    print(f"临床主题：{main_topic}")
    print(f"主要结局变量：{outcome_name}")
    print("-" * 70)
    print("配置文件：configs/app.yaml 读取成功")
    print("正式日志系统：初始化成功")
    print("路径管理系统：初始化成功")
    print(f"正式日志文件：{log_file}")
    print(f"数据目录：{paths_summary['data_dir']}")
    print(f"输出目录：{paths_summary['output_dir']}")
    print(f"模型目录：{paths_summary['model_dir']}")
    print(f"报告目录：{paths_summary['report_dir']}")
    print("启动检查文件：startup_check.txt 写入成功")
    print("=" * 70)


def main() -> None:
    """
    Python 主程序入口。

    当前执行顺序：
    1. 获取项目根目录
    2. 设置 Python 模块路径
    3. 读取配置文件
    4. 初始化正式日志系统
    5. 初始化路径管理系统
    6. 自动确认常用文件夹存在
    7. 输出启动摘要
    8. 写入启动检查文件
    """
    project_root = get_project_root()
    setup_python_path(project_root)

    config = load_project_config()

    from bggdeep.core.logger import setup_logger
    from bggdeep.core.paths import get_project_paths, ensure_project_directories

    logger = setup_logger(config)

    logger.info("BggDeepLearning 主程序开始启动")
    logger.info("项目根目录：%s", project_root)

    project_paths = get_project_paths(config)
    ensure_project_directories(project_paths, logger=logger)

    paths_summary = {
        "data_dir": project_paths.data_dir,
        "output_dir": project_paths.output_dir,
        "log_dir": project_paths.log_dir,
        "model_dir": project_paths.model_dir,
        "report_dir": project_paths.report_dir,
    }

    print_startup_summary(project_root, config, paths_summary)
    write_startup_log(project_root, config, project_paths.log_dir)

    logger.info("路径管理系统初始化完成")
    logger.info("数据目录：%s", project_paths.data_dir)
    logger.info("输出目录：%s", project_paths.output_dir)
    logger.info("模型目录：%s", project_paths.model_dir)
    logger.info("报告目录：%s", project_paths.report_dir)
    logger.info("主程序启动流程执行完成")
    logger.info("当前项目骨架运行正常")


if __name__ == "__main__":
    main()