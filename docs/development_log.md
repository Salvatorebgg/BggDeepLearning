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
