# -*- coding: utf-8 -*-

"""
BggDeepLearning FastAPI 应用入口

File:
apps/api/main.py

Purpose:
- FastAPI app 实例创建和生命周期管理
- CORS 配置
- 路由注册
- 启动/关闭事件

Usage:
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
python apps/api/main.py
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path


def find_project_root() -> Path:
    """
    Find project root by locating configs/app.yaml.
    """
    current_path = Path(__file__).resolve()

    for parent in [current_path, *current_path.parents]:
        if (parent / "configs" / "app.yaml").exists():
            return parent

    raise FileNotFoundError(
        "Project root was not found. configs/app.yaml is missing."
    )


def setup_project() -> Path:
    """
    Set up project Python path and return project root.
    """
    project_root = find_project_root()
    python_dir = project_root / "python"

    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))

    return project_root


@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan context manager.

    - Startup: Validate that models and preprocessor are available
    - Shutdown: Clean up resources
    """
    project_root = setup_project()

    # Validate critical files on startup
    from bggdeep.core.logger import setup_logger
    from bggdeep.core.config import load_yaml_config

    config = load_yaml_config("app.yaml")
    logger = setup_logger(config)

    model_dir = project_root / "outputs" / "models"
    preprocessor_file = model_dir / "tabular_preprocessor.joblib"

    missing = []
    if not preprocessor_file.exists():
        missing.append(str(preprocessor_file))

    if missing:
        logger.warning("API startup: 以下文件缺失，部分接口可能不可用：")
        for path in missing:
            logger.warning("  - %s", path)
    else:
        logger.info("API startup: 模型文件和预处理器检查通过")

    # Store project root in app state
    app.state.project_root = project_root
    app.state.logger = logger

    # Pre-warm the engine
    try:
        from bggdeep.api.engine import InferenceEngine, create_engine

        engine = create_engine(
            model_dir=model_dir,
            preprocessor_file=preprocessor_file,
        )
        app.state.engine = engine
        logger.info("API startup: 推理引擎初始化完成")
    except Exception as exc:
        logger.warning("API startup: 推理引擎初始化失败 — %s", exc)
        app.state.engine = None

    logger.info("BggDeepLearning FastAPI 服务启动完成")

    yield

    # Shutdown
    logger.info("BggDeepLearning FastAPI 服务已关闭")


# Create FastAPI app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Set up project path before anything else
_project_root = setup_project()

app = FastAPI(
    title="BggDeepLearning Clinical Risk Prediction API",
    description="""
    BggDeepLearning 临床风险预测 API 服务。

    ## 功能

    - **单个患者预测**：传入临床特征，返回 poor_outcome 风险概率
    - **批量预测**：批量传入患者数据，返回批量预测结果
    - **DOCX 报告导出**：预测 + 生成 Word 报告
    - **健康检查**：检查服务状态

    ## 可用模型

    | 模型键 | 中文名称 |
    |--------|----------|
    | `gradient_boosting` | 梯度提升模型 |
    | `random_forest` | 随机森林 |
    | `logistic` | 逻辑回归 |
    | `deep_mlp` | 深度神经网络 (PyTorch MLP) |

    ## 临床警告

    **当前模型基于模拟数据训练，仅用于工程演示和研究流程搭建，不能用于真实临床诊疗决策。**
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
from bggdeep.api.routes import router

app.include_router(router)


# Serve auto-generated OpenAPI docs at /docs and /redoc


# ─────────────────────────────────────────────────────────
# Direct run entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Use the direct string path with the current file's absolute path
    # so uvicorn can reload properly
    app_path = f"{Path(__file__).stem}:app"
    print("=" * 60)
    print("BggDeepLearning FastAPI 服务启动")
    print(f"项目根目录: {_project_root}")
    print("API 文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/api/v1/health")
    print("=" * 60)

    uvicorn.run(
        app_path,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
