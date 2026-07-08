# BggDeepLearning

## 项目简介

**BggDeepLearning** 是一个从零开始构建的、面向临床研究方向的多语言多模态深度学习与智能决策支持平台。

本项目不是一个简单 Demo，而是一个长期迭代的大型工程项目。项目将逐步包含临床数据管理、数据清洗、表格数据建模、医学影像建模、医学文本建模、临床统计分析、模型评估、可解释性分析、后端 API、前端可视化、数据库管理、自动化报告生成、临床仿真和工程化部署等模块。

## 项目定位

本项目定位为：

> 面向临床研究的多语言多模态深度学习与智能决策支持平台。

推荐英文描述：

> A multi-language multimodal deep learning platform for clinical research and decision support.

## 当前主线方向

当前项目主线为：

> 重症 / 急诊 / 围手术期患者不良结局多模态风险预测平台。

早期阶段先使用模拟临床表格数据作为基础，主要结局变量为：

```text
poor_outcome
```

含义：

```text
0 = 未发生严重不良临床结局
1 = 发生严重不良临床结局
```

后续将逐步扩展到：

1. ICU 死亡风险预测；
2. 休克风险预测；
3. 大出血风险预测；
4. 围手术期并发症预测；
5. 医学影像辅助诊断；
6. 临床文本挖掘；
7. 多模态临床决策支持。

## 技术路线

项目采用多语言工程架构：

| 技术类型                    | 主要用途                             |
| ----------------------- | -------------------------------- |
| Python                  | 项目主流程、数据处理、机器学习、深度学习、后端 API      |
| R                       | 临床统计分析、Table 1、生存分析、DCA、nomogram |
| C                       | 底层数学函数、基础高性能计算                   |
| C++                     | 高性能模拟、图像处理加速、风险计算引擎              |
| SQL                     | 数据库建表、数据管理                       |
| PowerShell / Batch      | Windows 自动化脚本                    |
| TypeScript / JavaScript | 前端交互界面                           |
| YAML / TOML / JSON      | 配置管理                             |
| Dockerfile              | 后期容器化部署                          |
| CMake                   | C / C++ 工程构建                     |

## 当前项目阶段

当前处于：

```text
阶段 1：多语言项目骨架初始化
```

目前已经完成：

1. 创建项目根目录；
2. 创建基础文件夹结构；
3. 创建 Python 包结构；
4. 创建第一个 Python 主程序；
5. 创建项目配置文件 `configs/app.yaml`；
6. 创建配置读取模块 `config.py`；
7. 创建正式日志模块 `logger.py`；
8. 创建路径管理模块 `paths.py`；
9. 创建 Python 虚拟环境 `.venv`；
10. 创建依赖文件 `requirements.txt`；
11. 创建 Git 忽略文件 `.gitignore`；
12. 创建项目说明文档 `README.md`；
13. 创建开发日志 `docs/development_log.md`。

## 项目目录结构

当前基础目录结构如下：

```text
BggDeepLearning
├─ .venv
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ configs
│  └─ app.yaml
├─ python
│  └─ bggdeep
│     ├─ __init__.py
│     ├─ main.py
│     ├─ core
│     │  ├─ __init__.py
│     │  ├─ config.py
│     │  ├─ logger.py
│     │  └─ paths.py
│     ├─ data
│     │  └─ __init__.py
│     ├─ models
│     │  └─ __init__.py
│     ├─ evaluation
│     │  └─ __init__.py
│     └─ utils
│        └─ __init__.py
├─ r
├─ cpp
├─ c
├─ database
├─ scripts
├─ data
├─ outputs
├─ docs
│  └─ development_log.md
└─ tests
```

## 当前运行方式

### 1. 进入项目目录

```powershell
cd "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"
```

### 2. 激活虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
```

激活成功后，终端前面应显示：

```text
(.venv)
```

### 3. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 4. 运行主程序

```powershell
python python\bggdeep\main.py
```

运行成功后，终端应显示：

```text
BggDeepLearning Python 主程序启动成功
配置文件：configs/app.yaml 读取成功
正式日志系统：初始化成功
路径管理系统：初始化成功
```

## 当前已实现功能

### 1. 配置文件读取

配置文件位置：

```text
configs/app.yaml
```

配置读取模块：

```text
python/bggdeep/core/config.py
```

该模块可以读取项目名称、版本、路径、临床主题、日志配置等信息。

### 2. 日志系统

日志模块位置：

```text
python/bggdeep/core/logger.py
```

正式日志文件输出到：

```text
outputs/logs/bggdeep.log
```

### 3. 路径管理系统

路径模块位置：

```text
python/bggdeep/core/paths.py
```

该模块统一管理：

1. 数据路径；
2. 日志路径；
3. 图片路径；
4. 报告路径；
5. 模型路径；
6. 表格路径；
7. 预测结果路径。

### 4. 项目主入口

主程序位置：

```text
python/bggdeep/main.py
```

运行命令：

```powershell
python python\bggdeep\main.py
```

## 当前 Python 依赖

当前基础依赖文件为：

```text
requirements.txt
```

当前主要依赖包括：

1. PyYAML；
2. pytest；
3. numpy；
4. pandas；
5. scikit-learn；
6. matplotlib。

后续会逐步增加：

1. PyTorch；
2. XGBoost；
3. LightGBM；
4. SHAP；
5. FastAPI；
6. python-docx；
7. OpenCV；
8. Transformers。

## 开发原则

本项目长期遵守以下原则：

1. 先跑通，再复杂；
2. 每一步只做少量可验证内容；
3. 每个模块都要能单独运行；
4. 所有路径统一管理；
5. 所有关键步骤写日志；
6. 所有依赖进入虚拟环境；
7. 不把真实临床数据提交到 Git；
8. 所有输出统一放入 `outputs`；
9. 重要功能必须有测试；
10. 每一步开发都更新开发日志。

## 后续开发路线

近期开发计划：

1. 创建基础文档；
2. 创建 R 环境检查脚本；
3. 创建 C / C++ 最小代码结构；
4. 创建 SQL 建表脚本；
5. 创建第一个临床数据模拟模块；
6. 生成第一份模拟临床数据；
7. 创建数据清洗模块；
8. 训练第一个 Logistic 回归模型；
9. 绘制第一张 ROC 曲线；
10. 创建第一个 PyTorch 表格深度学习模型。

## 注意事项

### 1. 必须使用虚拟环境

每次打开项目后，建议先激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

确认终端前面出现：

```text
(.venv)
```

### 2. 不要提交 `.venv`

`.venv` 是本地虚拟环境，不应该提交到 Git。

### 3. 不要提交真实临床数据

真实临床数据应放在：

```text
data/raw
```

并且默认不提交到 Git。

### 4. 日志文件不提交

日志文件会自动生成在：

```text
outputs/logs
```

默认不进入版本管理。

## 项目状态

当前状态：

```text
项目骨架可以正常启动。
配置文件可以正常读取。
日志系统可以正常写入。
路径管理系统可以正常工作。
```
