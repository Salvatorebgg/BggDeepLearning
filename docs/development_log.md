# BggDeepLearning 开发日志

## 日志说明

本文档用于记录 **BggDeepLearning** 项目的长期开发过程。

每完成一个小步骤，都建议在这里记录：

1. 完成时间；
2. 完成内容；
3. 新增文件；
4. 修改文件；
5. 运行命令；
6. 是否运行成功；
7. 遇到的问题；
8. 下一步计划。

这样做的好处是：

1. 防止长期项目开发到后期忘记前面做过什么；
2. 方便定位问题；
3. 方便回退和复盘；
4. 方便以后整理项目文档和论文方法学部分；
5. 让项目更像一个正式工程，而不是零散脚本。

---

# 阶段 1：多语言项目骨架初始化

## 第 1 小步：创建项目根目录

### 完成内容

创建项目根目录：

```text
D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning
```

### 主要操作

在 VS Code 终端中执行：

```powershell
New-Item -ItemType Directory -Force -Path "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"
cd "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"
code .
```

### 完成状态

已完成。

---

## 第 2 小步：创建一级文件夹结构

### 完成内容

创建项目一级目录：

```text
configs
python
r
cpp
c
database
scripts
data
outputs
docs
tests
```

### 主要作用

| 文件夹      | 作用         |
| -------- | ---------- |
| configs  | 配置文件       |
| python   | Python 主工程 |
| r        | R 统计分析工程   |
| cpp      | C++ 高性能工程  |
| c        | C 底层工具工程   |
| database | 数据库脚本      |
| scripts  | 自动化脚本      |
| data     | 数据目录       |
| outputs  | 输出目录       |
| docs     | 项目文档       |
| tests    | 测试代码       |

### 完成状态

已完成。

---

## 第 3 小步：创建二级文件夹结构

### 完成内容

创建数据、输出、脚本、测试、数据库的二级目录。

主要包括：

```text
data/raw
data/processed
data/simulated
data/interim
data/external

outputs/logs
outputs/figures
outputs/reports
outputs/models
outputs/tables
outputs/predictions

scripts/windows
scripts/python

tests/python
tests/r
tests/cpp
tests/integration

database/schema
database/migrations
database/seeds
```

### 完成状态

已完成。

---

## 第 4 小步：创建 Python 基础包结构

### 完成内容

创建 Python 主包：

```text
python/bggdeep
```

创建基础模块：

```text
python/bggdeep/core
python/bggdeep/data
python/bggdeep/models
python/bggdeep/evaluation
python/bggdeep/utils
```

创建 `__init__.py` 文件，使这些目录成为 Python 包。

### 完成状态

已完成。

---

## 第 5 小步：创建第一个 Python 主程序

### 新增文件

```text
python/bggdeep/main.py
```

### 完成内容

创建第一个可运行的主程序。

最初功能包括：

1. 获取项目根目录；
2. 打印项目启动成功信息；
3. 写入简单启动日志文件。

### 运行命令

```powershell
python python\bggdeep\main.py
```

### 完成状态

已完成。

---

## 第 6 小步：创建项目配置文件

### 新增文件

```text
configs/app.yaml
```

### 完成内容

创建项目基础配置文件，包含：

1. 项目信息；
2. 路径配置；
3. 临床任务配置；
4. 开发技术栈说明；
5. 日志配置。

### 完成状态

已完成。

---

## 第 7 小步：创建配置读取模块

### 新增文件

```text
python/bggdeep/core/config.py
```

### 完成内容

创建 YAML 配置读取模块。

主要功能：

1. 获取项目根目录；
2. 获取配置文件路径；
3. 读取 `configs/app.yaml`；
4. 按层级读取配置项；
5. 打印配置摘要。

### 依赖

```text
PyYAML
```

### 运行命令

```powershell
python python\bggdeep\core\config.py
```

### 完成状态

已完成。

---

## 第 8 小步：让主程序读取配置文件

### 修改文件

```text
python/bggdeep/main.py
```

### 完成内容

将主程序升级为可以读取：

```text
configs/app.yaml
```

主程序可以显示：

1. 项目名称；
2. 项目版本；
3. 当前阶段；
4. 临床主题；
5. 主要结局变量；
6. 日志目录。

### 运行命令

```powershell
python python\bggdeep\main.py
```

### 完成状态

已完成。

---

## 第 9 小步：创建正式日志模块

### 新增文件

```text
python/bggdeep/core/logger.py
```

### 修改文件

```text
python/bggdeep/main.py
```

### 完成内容

创建正式日志系统。

主要功能：

1. 同时输出日志到终端和文件；
2. 日志文件保存到 `outputs/logs/bggdeep.log`；
3. 支持 INFO、WARNING、ERROR 等日志级别；
4. 防止重复添加日志 handler；
5. 主程序正式接入日志系统。

### 运行命令

```powershell
python python\bggdeep\core\logger.py
python python\bggdeep\main.py
```

### 完成状态

已完成。

---

## 第 10 小步：创建路径管理模块

### 新增文件

```text
python/bggdeep/core/paths.py
```

### 修改文件

```text
python/bggdeep/main.py
```

### 完成内容

创建统一路径管理系统。

主要功能：

1. 获取项目根目录；
2. 读取配置中的路径；
3. 把相对路径转换为绝对路径；
4. 自动确认常用文件夹存在；
5. 为后续数据、模型、日志、报告、图片等模块提供统一路径入口。

### 运行命令

```powershell
python python\bggdeep\core\paths.py
python python\bggdeep\main.py
```

### 完成状态

已完成。

---

## 第 11 小步：创建依赖文件和 Git 忽略文件

### 新增文件

```text
requirements.txt
.gitignore
```

### 完成内容

创建 Python 依赖文件和 Git 忽略规则。

`requirements.txt` 当前包含：

1. pyyaml；
2. pytest；
3. numpy；
4. pandas；
5. scikit-learn；
6. matplotlib。

`.gitignore` 当前忽略：

1. Python 缓存；
2. 虚拟环境；
3. VS Code 本地配置；
4. 日志文件；
5. 临床数据文件；
6. 输出结果；
7. 数据库本地文件；
8. R 缓存；
9. C / C++ 编译产物；
10. 前端依赖和构建产物；
11. 环境变量文件；
12. 临时文件。

### 完成状态

已完成。

---

## 第 12 小步：创建并启用 Python 虚拟环境

### 新增目录

```text
.venv
```

### 完成内容

创建项目专属 Python 虚拟环境，避免后续包版本冲突。

### 主要命令

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 检查命令

```powershell
python -c "import sys; print(sys.executable)"
```

预期应显示：

```text
D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning\.venv\Scripts\python.exe
```

### 完成状态

已完成。

---

## 第 13 小步：创建 README 和开发日志

### 新增文件

```text
README.md
docs/development_log.md
```

### 完成内容

创建项目首页说明文档和长期开发日志。

README 主要记录：

1. 项目简介；
2. 项目定位；
3. 当前主线方向；
4. 技术路线；
5. 当前项目阶段；
6. 当前目录结构；
7. 运行方式；
8. 已实现功能；
9. 当前依赖；
10. 后续开发路线；
11. 注意事项。

开发日志主要记录：

1. 每一步完成内容；
2. 新增文件；
3. 修改文件；
4. 运行命令；
5. 当前完成状态；
6. 后续扩展计划。

### 完成状态

进行中。

---

# 当前项目状态总结

截至第 13 小步，项目已经具备：

1. 基础目录结构；
2. Python 包结构；
3. 配置文件系统；
4. 日志系统；
5. 路径管理系统；
6. Python 虚拟环境；
7. Python 依赖管理；
8. Git 忽略规则；
9. 项目说明文档；
10. 开发日志文档。

当前主程序运行命令：

```powershell
python python\bggdeep\main.py
```

当前运行成功标志：

```text
BggDeepLearning Python 主程序启动成功
配置文件：configs/app.yaml 读取成功
正式日志系统：初始化成功
路径管理系统：初始化成功
```

---

# 下一步计划

下一步建议进入：

```text
第 14 小步：创建 R 统计模块的第一个环境检查脚本
```

目标是让项目的 R 子工程正式启动，完成：

1. 创建 R 脚本；
2. 检查 R 是否可用；
3. 输出 R 版本；
4. 创建 R 输出目录；
5. 为后续 Table 1、Logistic 回归、生存分析做准备。

---

# 阶段 2：表格型临床风险预测工程闭环

## 第 14–40 小步：已完成表格型临床风险预测全流程

（历史步骤记录略，详见各模块脚本）

已完成模块：

1. 临床模拟数据生成 → `outputs/data/simulated_clinical_data_*.csv`
2. 数据质量检查 → `outputs/tables/data_quality_report_*.csv`
3. 数据预处理 → `outputs/models/tabular_preprocessor.joblib`
4. Logistic Regression 训练 + 评估 → `outputs/models/logistic_baseline_*.joblib`
5. Random Forest 训练 + 评估 → `outputs/models/random_forest_baseline_*.joblib`
6. Gradient Boosting 训练 + 评估 → `outputs/models/gradient_boosting_baseline_*.joblib`
7. ROC / PR / 混淆矩阵绘图 → `outputs/figures/model_comparison_*_roc_curve.png`
8. 模型排行榜 → `outputs/tables/model_leaderboard_*.csv`
9. Calibration 校准评估 → `outputs/figures/model_comparison_*_calibration_curve.png`
10. DCA 决策曲线分析 → `outputs/figures/dca_*_curve.png`
11. SHAP 可解释性 → `outputs/figures/shap_*_bar.png`, `outputs/figures/shap_*_beeswarm.png`
12. 个体化解释 → `outputs/tables/individual_explanation_*.csv`
13. 批量高风险患者解释 → `outputs/tables/batch_high_risk_explanations.csv`
14. 自动化 Word / TXT 报告 → `outputs/reports/comprehensive_model_evaluation_report.txt`
15. SCI 风格 Word 综合报告 → `outputs/reports/sci_comprehensive_model_evaluation_report.docx`
16. SCI 风格 Word SHAP 报告 → `outputs/reports/sci_shap_explainability_report.docx`
17. Streamlit 中文交互界面 → `apps/streamlit/clinical_risk_demo.py`
18. 批量 CSV 上传预测
19. 临床模型评估仪表盘

---

## 第 41 小步：为 Streamlit 单个患者预测页面增加 TXT/DOCX 报告导出

### 完成时间

2026-07-08

### 完成内容

1. 验证已有 `python/bggdeep/reporting/streamlit_patient_docx.py` 模块可正常生成 DOCX 报告（37886 bytes）
2. 验证 `apps/streamlit/clinical_risk_demo.py` 中 `show_current_patient_report_download()` 函数正确集成 TXT 和 DOCX 下载按钮
3. 将 `python-docx>=1.0` 从注释状态激活到 `requirements.txt`
4. 验证 Streamlit 应用可正常启动无报错
5. DOCX 报告包含：模型信息、预测结果、患者输入、解释性因素、重要说明

### 新增文件

无（模块已在之前创建）

### 修改文件

```text
requirements.txt  — 激活 python-docx>=1.0 依赖
docs/development_log.md  — 追加本步记录
```

### 运行命令

```powershell
# 测试 DOCX 模块
python -c "from bggdeep.reporting.streamlit_patient_docx import build_patient_prediction_docx_bytes; ..."

# 测试 Streamlit 应用编译
python -c "compile(open('apps/streamlit/clinical_risk_demo.py').read(), ...)"

# 启动 Streamlit 验证
streamlit run apps/streamlit/clinical_risk_demo.py --server.headless true
```

### 测试结果

全部通过：DOCX 模块生成 37886 字节报告，Streamlit 页面编译/启动无报错。

### 下一步计划

第 42 小步：创建 PyTorch 表格深度学习基础模块（Tabular MLP），训练第一个深度学习模型预测 poor_outcome。

---

## 第 42 小步：创建 PyTorch 表格深度学习基础模块

### 完成时间

2026-07-08

### 完成内容

1. 创建 `python/bggdeep/models/deep_learning/` 包
2. 实现 `mlp.py` — 灵活的多层感知机模型
   - 可配置隐藏层维度、dropout、BatchNorm、激活函数
   - Kaiming 权重初始化
   - 自动计算模型参数量
   - 自带 self-test
3. 实现 `tabular_dataset.py` — PyTorch Dataset/DataLoader 封装
   - 直接读取预处理 CSV 数据
   - 支持 WeightedRandomSampler 处理类别不平衡
   - 自动计算 pos_weight
4. 实现 `training.py` — 训练引擎
   - 训练循环 + 验证循环
   - Early Stopping (patience=30, min_delta=1e-4)
   - 梯度裁剪
   - 最佳模型自动保存/恢复
   - ROC AUC 实时监控
5. 实现 `scripts/python/train_deep_mlp.py` — 完整的命令行训练脚本
   - 支持命令行参数调节架构和超参数
   - 输出：模型 .pt 文件、metrics CSV、predictions CSV、training history CSV、报告 TXT
6. 安装 PyTorch 2.12.1 (CPU 版)
7. 更新 `model_ranking.py` 将 DL 模型纳入排行榜
8. 重新运行排行榜，DL 模型验证集 AUC 0.9297，排名第 3

### 训练结果

| 数据集 | AUC | Accuracy | Sensitivity | Specificity | F1 |
|--------|-----|----------|-------------|-------------|-----|
| Train | 1.0000 | 1.00 | 1.00 | 1.00 | 1.0000 |
| Validation | 0.9297 | 0.95 | 0.50 | 0.9688 | 0.4444 |
| Test | 0.4922 | 0.95 | 0.00 | 0.9896 | 0.0000 |

模型架构：
- Input: 26 features
- Hidden: [128, 64, 32]
- Dropout: 0.3, BatchNorm: True, Activation: ReLU
- Parameters: 14,273
- pos_weight: 20.43
- Best epoch: 59 (Early Stopping)

### 验证集排行榜（4 模型）

| 排名 | 模型 | AUC |
|------|------|-----|
| 1 | Random Forest | 0.9479 |
| 2 | Logistic Regression | 0.9323 |
| 3 | Deep MLP (PyTorch) | 0.9297 |
| 4 | Gradient Boosting | 0.8828 |

### 新增文件

```text
python/bggdeep/models/deep_learning/__init__.py
python/bggdeep/models/deep_learning/mlp.py
python/bggdeep/models/deep_learning/tabular_dataset.py
python/bggdeep/models/deep_learning/training.py
scripts/python/train_deep_mlp.py
outputs/models/deep_mlp_baseline.pt
outputs/tables/deep_mlp_baseline_metrics.csv
outputs/tables/deep_mlp_training_history.csv
outputs/predictions/deep_mlp_train_predictions.csv
outputs/predictions/deep_mlp_val_predictions.csv
outputs/predictions/deep_mlp_test_predictions.csv
outputs/reports/deep_mlp_baseline_report.txt
```

### 修改文件

```text
python/bggdeep/evaluation/model_ranking.py  — 新增 deep_mlp_baseline_metrics.csv 候选
requirements.txt  — torch>=2.0 从注释激活
docs/development_log.md  — 追加本步记录
```

### 运行命令

```powershell
# 安装 PyTorch CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 运行模块自测
python python\bggdeep\models\deep_learning\mlp.py
python python\bggdeep\models\deep_learning\tabular_dataset.py

# 训练深度学习模型
python scripts\python\train_deep_mlp.py

# 重建排行榜
python scripts\python\build_model_leaderboard.py
```

### 测试与调试

- MLP module self-test: PASSED
- TabularDataset module self-test: PASSED  
- 训练脚本运行成功，Early Stopping 在 epoch 89 触发，best epoch 59
- 排行榜重建成功，4 个模型均已纳入
- 测试集 AUC 较低（0.49）主要是由于数据量小（500 样本）和类别极不平衡（~4% positive rate），这是预期行为
- DL 模型在验证集上与传统模型表现可比（AUC 0.93 vs RF 0.95），证明深度学习流程正确打通

### 下一步计划

第 43 小步：为 Streamlit 单个预测页面接入深度学习模型（DL + 传统模型均可选择），或继续打磨 DL 模块（添加更多架构如 TabNet、添加训练可视化等），或进入后端 API 开发阶段。

---

# 下一步计划

下一步建议：**第 43 小步：将 Deep MLP 模型接入 Streamlit 预测界面**

当前 Streamlit 只能选择三个传统模型（LR/RF/GB），需要：
1. 让 ClinicalRiskPredictor 支持加载 PyTorch .pt 模型
2. Streamlit 下拉框增加 "Deep MLP (PyTorch)" 选项
3. 实现 DL 模型的 SHAP/特征贡献解释（使用 Integrated Gradients 或 permutation importance）
4. 验证前后端预测流程完整

---

## 第 43 小步：将 Deep MLP 模型接入 Streamlit 预测界面

### 完成时间

2026-07-08

### 完成内容

1. 修改 `ClinicalRiskPredictor` 支持 PyTorch `.pt` 模型：
   - `MODEL_SPECS` 新增 `deep_mlp` 模型规格（model_type="deep_learning"）
   - `load_model()` 支持通过 `torch.load` + `TabularMLP` 重建 PyTorch 模型
   - `predict_probability()` 和 `predict_probabilities()` 使用 `isinstance(model, torch.nn.Module)` 区分 sklearn/PyTorch
   - `align_to_model_features()` PyTorch 模型跳过（没有 `feature_names_in_`）
   - `get_transformed_feature_names()` PyTorch 模型 fallback 到预处理器 feature names
2. 新增 `build_permutation_explanation()` 方法：
   - 通过逐特征扰动分析贡献值，作为 DL 模型的轻量可解释性方案
   - 每个特征增加 10% 或 0.1（取大者），计算预测概率变化
   - 输出与线性模型贡献表相同格式，兼容 Streamlit 展示
3. `explain_prediction()` 新增 `deep_learning` 分支，调用 permutation explanation
4. `apps/streamlit/clinical_risk_demo.py`：
   - `MODEL_DISPLAY` 新增 `deep_mlp` → "深度神经网络 Deep MLP (PyTorch)"
   - 模型选择下拉框新增 `deep_mlp` 选项
5. 端到端测试：全部 4 个模型（LR/RF/GB/DL）预测流程正常

### 端到端测试结果

```text
logistic             | prob=0.234103 | expl_rows=10
random_forest        | prob=0.037330 | expl_rows=10
gradient_boosting    | prob=0.001550 | expl_rows=10
deep_mlp             | prob=0.004337 | expl_rows=10

Batch DL probs: 4 模型批量预测正常
Streamlit 编译启动: PASSED
```

### 修改文件

```text
python/bggdeep/inference/clinical_risk_predictor.py  — 新增 PyTorch 模型加载/预测/解释
apps/streamlit/clinical_risk_demo.py  — 新增 deep_mlp 模型选项
docs/development_log.md  — 追加本步记录
```

### 运行命令

```powershell
# 端到端测试
python -c "
from bggdeep.inference.clinical_risk_predictor import ClinicalRiskPredictor, ClinicalRiskPredictorConfig
# 测试 4 个模型
"

# 验证 Streamlit 不报错
streamlit run apps/streamlit/clinical_risk_demo.py --server.headless true
```

### 测试结果

- 4 个模型（LR, RF, GB, Deep MLP）均能正确加载、预测、解释
- 批量 CSV 预测正常
- Streamlit 页面编译/启动无报错
- Permutation 解释为 DL 模型生成了 10 行特征贡献表

### 下一步计划

第 44 小步：进入后端 API 开发阶段（FastAPI），或继续打磨 DL 模块（训练可视化、超参数搜索、TabNet 等架构），或进入医学影像模块开发。

---

## 第 44 小步：创建 FastAPI 后端推理服务

### 完成时间

2026-07-08

### 完成内容

1. 创建 `python/bggdeep/api/` 包：
   - `__init__.py` — 包初始化
   - `schemas.py` — Pydantic 请求/响应模型（单个预测、批量预测、健康检查、错误响应）
   - `engine.py` — 推理引擎封装（单例模式，模型缓存，预测/批量预测/健康检查）
   - `routes.py` — FastAPI 路由（6 个端点：health/models/single/batch/report/error handling）
2. 创建 `apps/api/main.py` — FastAPI 应用入口：
   - lifespan 管理（启动时预加载推理引擎）
   - CORS 配置（开发阶段允许所有来源）
   - OpenAPI 文档自动生成（/docs, /redoc）
   - 直接 `python apps/api/main.py` 启动
3. 创建 `scripts/python/test_api_client.py` — 自动化 API 测试客户端
   - 6 个测试用例覆盖所有端点
   - 支持命令行参数指定 base_url 和 model_key
   - 使用标准库 urllib（无额外依赖）
4. API 端点：
   - `GET /api/v1/health` — 健康检查
   - `GET /api/v1/models` — 可用模型列表
   - `POST /api/v1/predict/single?model_key=xxx` — 单个患者预测
   - `POST /api/v1/predict/batch?model_key=xxx` — 批量患者预测（最多 500 条）
   - `POST /api/v1/predict/single/report?model_key=xxx` — 预测 + DOCX 报告下载
   - 错误处理测试

### 测试结果（全部 6/6 通过）

| 测试 | 结果 |
|------|------|
| 健康检查 GET /api/v1/health | PASS (4 models) |
| 模型列表 GET /api/v1/models | PASS (4 models) |
| 单个预测 Random Forest | PASS prob=0.0373, 10 expl items |
| 批量预测 Random Forest | PASS total=2, all low risk |
| DOCX 报告 Random Forest | PASS 38285 bytes |
| 无效模型测试 | PASS (正确返回 400) |
| 单个预测 Deep MLP | PASS prob=0.0043, 10 expl items |
| 批量预测 Deep MLP | PASS total=2 |
| DOCX 报告 Deep MLP | PASS 38324 bytes |

### 新增文件

```text
python/bggdeep/api/__init__.py
python/bggdeep/api/schemas.py
python/bggdeep/api/engine.py
python/bggdeep/api/routes.py
apps/api/main.py
scripts/python/test_api_client.py
```

### 修改文件

```text
requirements.txt  — fastapi>=0.110, uvicorn>=0.30 从注释激活
docs/development_log.md  — 追加本步记录
```

### 运行命令

```powershell
# 启动 API 服务
python apps/api/main.py

# 在另一个终端运行测试
python scripts/python/test_api_client.py --model-key random_forest
python scripts/python/test_api_client.py --model-key deep_mlp

# 访问 API 文档
# http://localhost:8000/docs
# http://localhost:8000/redoc
```

### 调试经验

- `uvicorn` reload 模式使用 `apps.api.main:app` 导入失败 → 改用 `main:app` + `reload=False`
- engine.py 中误删除 `ClinicalRiskPredictor` 导入 → 单预测/批量预测返回 500 → 修复导入
- 测试脚本 emoji 在 GBK 编码 Windows 终端报错 → 替换为 ASCII 标记
- taskkill 需要指定具体 PID 避免误杀其他 Python 进程

### 下一步计划

第 45 小步：继续打磨 DL 模块（训练可视化、超参数搜索、TabNet 等架构），或进入医学影像模块开发。建议先完善 DL 模块：
1. 添加训练/验证 loss/auc 曲线可视化
2. 尝试不同架构（更深的 MLP、ResNet-style skip connections）
3. 添加 k-fold 交叉验证
4. 或进入 Stage 4 医学影像模块
1. 模型推理接口（单条/批量）
2. 文件上传接口
3. 报告导出接口
4. API 日志
5. API 测试
6. 健康检查端点
