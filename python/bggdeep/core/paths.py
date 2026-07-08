"""
BggDeepLearning 路径管理模块

文件位置：
python/bggdeep/core/paths.py

作用：
1. 获取项目根目录
2. 根据 configs/app.yaml 读取项目常用路径
3. 把相对路径转换为绝对路径
4. 自动创建必要文件夹
5. 为后续数据、模型、日志、报告、图片等模块提供统一路径入口

为什么需要这个模块？

大型项目中，路径非常容易混乱。
例如：
- 数据保存在哪里？
- 日志保存在哪里？
- 模型保存在哪里？
- 图片保存在哪里？
- 报告保存在哪里？

如果每个文件都自己写路径，后期会非常难维护。

所以我们统一在这里管理路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import sys
import logging


@dataclass
class ProjectPaths:
    """
    项目路径集合。

    这个类用来集中保存项目中常用的路径。

    dataclass 是 Python 提供的一个工具，
    可以让我们更方便地定义这种“只保存数据”的类。
    """

    project_root: Path

    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    simulated_data_dir: Path

    output_dir: Path
    log_dir: Path
    figure_dir: Path
    report_dir: Path
    model_dir: Path
    table_dir: Path
    prediction_dir: Path

    config_dir: Path
    app_config_file: Path


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件位置：
    BggDeepLearning/python/bggdeep/core/paths.py

    层级关系：
    paths.py
    -> core
    -> bggdeep
    -> python
    -> BggDeepLearning

    所以 current_file.parents[3] 就是项目根目录。
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    return project_root


def setup_python_path(project_root: Optional[Path] = None) -> None:
    """
    把 BggDeepLearning/python 加入 Python 模块搜索路径。

    这样以后就可以正常导入：

    from bggdeep.core.config import load_yaml_config
    """
    if project_root is None:
        project_root = get_project_root()

    python_dir = project_root / "python"

    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    从配置字典中读取某个层级的值。

    示例：
    key_path = "paths.log_dir"

    等价于：
    config["paths"]["log_dir"]

    如果读取不到，就返回 default。
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


def resolve_project_path(project_root: Path, path_text: str) -> Path:
    """
    把配置文件中的路径转换为绝对路径。

    例如配置文件中写的是：

    outputs/logs

    这个函数会转换成：

    D:/BaiduSyncdisk/document/workdocument/task/BggDeepLearning/outputs/logs

    如果传入的本来就是绝对路径，则直接返回。
    """
    path = Path(path_text)

    if path.is_absolute():
        return path

    return project_root / path


def get_project_paths(config: Optional[Dict[str, Any]] = None) -> ProjectPaths:
    """
    获取项目所有常用路径。

    如果传入 config，就根据配置文件读取路径。
    如果不传入 config，就使用默认路径。
    """
    project_root = get_project_root()

    if config is None:
        config = {}

    data_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.data_dir", "data"),
    )

    raw_data_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.raw_data_dir", "data/raw"),
    )

    processed_data_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.processed_data_dir", "data/processed"),
    )

    simulated_data_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.simulated_data_dir", "data/simulated"),
    )

    output_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.output_dir", "outputs"),
    )

    log_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.log_dir", "outputs/logs"),
    )

    figure_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.figure_dir", "outputs/figures"),
    )

    report_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.report_dir", "outputs/reports"),
    )

    model_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.model_dir", "outputs/models"),
    )

    table_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.table_dir", "outputs/tables"),
    )

    prediction_dir = resolve_project_path(
        project_root,
        get_config_value(config, "paths.prediction_dir", "outputs/predictions"),
    )

    config_dir = project_root / "configs"
    app_config_file = config_dir / "app.yaml"

    return ProjectPaths(
        project_root=project_root,
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        simulated_data_dir=simulated_data_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        figure_dir=figure_dir,
        report_dir=report_dir,
        model_dir=model_dir,
        table_dir=table_dir,
        prediction_dir=prediction_dir,
        config_dir=config_dir,
        app_config_file=app_config_file,
    )


def ensure_project_directories(
    paths: ProjectPaths,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    自动创建项目运行所需的常用文件夹。

    如果文件夹已经存在，不会报错。

    参数：
    paths:
        ProjectPaths 对象，里面保存了所有路径。

    logger:
        日志对象。如果传入，就把创建信息写入日志。
    """
    directories = [
        paths.data_dir,
        paths.raw_data_dir,
        paths.processed_data_dir,
        paths.simulated_data_dir,
        paths.output_dir,
        paths.log_dir,
        paths.figure_dir,
        paths.report_dir,
        paths.model_dir,
        paths.table_dir,
        paths.prediction_dir,
        paths.config_dir,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

        if logger is not None:
            logger.info("确认文件夹存在：%s", directory)


def print_project_paths_summary(paths: ProjectPaths) -> None:
    """
    在终端打印项目路径摘要。
    """
    print("=" * 70)
    print("BggDeepLearning 项目路径检查成功")
    print("-" * 70)
    print(f"项目根目录：{paths.project_root}")
    print(f"配置目录：{paths.config_dir}")
    print(f"配置文件：{paths.app_config_file}")
    print("-" * 70)
    print(f"数据目录：{paths.data_dir}")
    print(f"原始数据目录：{paths.raw_data_dir}")
    print(f"处理后数据目录：{paths.processed_data_dir}")
    print(f"模拟数据目录：{paths.simulated_data_dir}")
    print("-" * 70)
    print(f"输出目录：{paths.output_dir}")
    print(f"日志目录：{paths.log_dir}")
    print(f"图片目录：{paths.figure_dir}")
    print(f"报告目录：{paths.report_dir}")
    print(f"模型目录：{paths.model_dir}")
    print(f"表格目录：{paths.table_dir}")
    print(f"预测结果目录：{paths.prediction_dir}")
    print("=" * 70)


def test_paths() -> None:
    """
    测试路径模块是否可用。

    运行方式：
    在项目根目录执行：

    python python\\bggdeep\\core\\paths.py
    """
    setup_python_path()

    try:
        from bggdeep.core.config import load_yaml_config

        config = load_yaml_config("app.yaml")
    except Exception as exc:
        print("配置文件读取失败，将使用默认路径配置。")
        print(f"错误信息：{exc}")
        config = None

    paths = get_project_paths(config)
    ensure_project_directories(paths)
    print_project_paths_summary(paths)


if __name__ == "__main__":
    test_paths()