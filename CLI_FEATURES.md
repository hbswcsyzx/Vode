# VODE CLI功能说明

## 当前CLI参数完整列表

### 主要参数

| 参数 | 类型 | 默认值 | 说明 | 是否必需 |
|------|------|--------|------|----------|
| `script` | 位置参数 | - | 要分析的Python脚本路径 | 是 |
| `script_args` | 位置参数 | - | 传递给脚本的参数 | 否 |

### 捕获模式参数

| 参数 | 类型 | 可选值 | 默认值 | 说明 |
|------|------|--------|--------|------|
| `--mode`, `-m` | 选项 | `static`, `dynamic` | `static` | 捕获模式 |
| `--stage4`, `-s4` | 标志 | - | False | 使用Stage 4管道（ExecutionNode-based） |

### 输出控制参数

| 参数 | 类型 | 可选值 | 默认值 | 说明 |
|------|------|--------|--------|------|
| `--format`, `-f` | 选项 | `svg`, `png`, `pdf`, `gv` | `svg` | 输出格式 |
| `--output`, `-o` | 选项 | 文件路径 | 自动生成 | 输出文件路径 |
| `--depth`, `-d` | 选项 | 整数 | 1 | 可视化的最大深度 |

### 其他参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--model-name` | 选项 | 要可视化的模型变量名 |
| `--collapse-loops` | 标志 | 折叠循环模式（默认True） |
| `--no-collapse-loops` | 标志 | 不折叠循环模式 |

## 功能组合矩阵

### 1. 捕获模式组合

| 组合 | 说明 | 示例 | 是否支持 |
|------|------|------|----------|
| `static` + `stage4` | Stage 4静态捕获 | `vode --stage4 --mode static script.py` | ✅ 支持 |
| `dynamic` + `stage4` | Stage 4动态捕获 | `vode --stage4 --mode dynamic script.py` | ❌ 需要样本输入（仅Python API） |
| `static` (无stage4) | 旧版静态捕获 | `vode --mode static script.py` | ✅ 支持（向后兼容） |
| `dynamic` (无stage4) | 旧版动态捕获 | `vode --mode dynamic script.py` | ❌ 需要样本输入（仅Python API） |

### 2. 输出格式组合

| 格式 | Stage 4 | 旧版 | 说明 |
|------|---------|------|------|
| `gv` | ✅ | ✅ | Graphviz DOT源文件 |
| `svg` | ✅ | ✅ | SVG矢量图（推荐） |
| `png` | ✅ | ✅ | PNG位图 |
| `pdf` | ✅ | ✅ | PDF文档 |

### 3. 深度控制组合

| 深度 | Stage 4 | 旧版 | 说明 |
|------|---------|------|------|
| `--depth 0` | ✅ | ✅ | 仅显示根节点 |
| `--depth 1` | ✅ | ✅ | 显示直接子节点 |
| `--depth 2+` | ✅ | ✅ | 显示更深层次 |
| 无depth参数 | 默认1 | 默认1 | 使用默认深度 |

## 功能详细说明

### Stage 4 vs 旧版管道

#### Stage 4管道（推荐）

- **特点：**
  - 基于ExecutionNode的新架构
  - 三列布局（INPUT | OPERATION | OUTPUT）
  - 更清晰的数据流可视化
  - 支持递归深度展开
  - 更好的性能

- **使用场景：**
  - 新项目开发
  - 需要清晰数据流的场景
  - 复杂模型分析

- **命令示例：**

  ```bash
  vode --stage4 --depth 1 script.py
  vode --stage4 --depth 2 --format png script.py
  ```

#### 旧版管道（向后兼容）

- **特点：**
  - 基于ComputationGraph
  - 传统节点-边图结构
  - 支持循环检测
  - 保持向后兼容性

- **使用场景：**
  - 旧项目维护
  - 需要循环检测的场景
  - 与旧版代码集成

- **命令示例：**

  ```bash
  vode --mode static --depth 5 script.py
  vode --mode static --format png --collapse-loops script.py
  ```

### 捕获模式详解

#### Static Mode（静态模式）

- **工作原理：** 分析模型结构，不运行forward pass
- **优点：**
  - 快速
  - 不需要样本输入
  - 内存占用小
- **缺点：**
  - 无运行时tensor信息
  - 无实际shape数据
- **适用场景：**
  - 快速查看模型结构
  - 参数统计
  - 模块层次分析

#### Dynamic Mode（动态模式）

- **工作原理：** 运行forward pass，捕获运行时信息
- **优点：**
  - 包含实际tensor shapes
  - 显示数据流
  - 捕获运行时行为
- **缺点：**
  - 需要样本输入
  - 较慢
  - 内存占用大
- **适用场景：**
  - 调试shape不匹配
  - 分析数据流
  - 验证模型行为
- **限制：** CLI模式下不支持（需要Python API提供样本输入）

### 深度控制详解

深度参数控制可视化的详细程度：

| 深度 | 显示内容 | 适用场景 |
|------|----------|----------|
| 0 | 仅根模块 | 超高层概览 |
| 1 | 主要组件 | 架构概览（推荐默认） |
| 2 | 子组件 | 详细结构分析 |
| 3 | 更深层次 | 深入分析 |
| 4+ | 最深层次 | 调试特定模块 |

**示例：**

```bash
# 快速概览
vode --stage4 --depth 0 model.py

# 标准分析
vode --stage4 --depth 1 model.py

# 详细分析
vode --stage4 --depth 2 model.py

# 深入调试
vode --stage4 --depth 3 model.py
```

## 子命令

### trace子命令

保存模型结构到JSON文件：

```bash
vode trace --output model.json script.py
```

### view子命令

可视化已保存的trace文件：

```bash
vode view model.json --format svg
```

## 常见使用模式

### 1. 快速分析

```bash
vode --stage4 script.py
```

- 使用默认参数
- Stage 4管道
- 深度1
- SVG输出

### 2. 详细分析

```bash
vode --stage4 --depth 2 --format png script.py
```

- 深度2查看更多细节
- PNG格式便于分享

### 3. 高分辨率输出

```bash
vode --stage4 --format pdf --output model.pdf script.py
```

- PDF格式
- 自定义输出路径

### 4. 传递脚本参数

```bash
vode --stage4 script.py --layers 5 --hidden 128
```

- 脚本参数跟在script.py后面

### 5. 向后兼容模式

```bash
vode --mode static --depth 5 --collapse-loops script.py
```

- 使用旧版管道
- 支持循环折叠

## 建议的参数简化方案

### 方案1：移除--stage4标志，设为默认行为

```bash
# 当前
vode --stage4 --depth 1 script.py

# 简化后
vode --depth 1 script.py

# 如需旧版，使用--legacy
vode --legacy --depth 1 script.py
```

### 方案2：合并mode和stage4

```bash
# 当前
vode --stage4 --mode static script.py

# 简化后（stage4成为默认）
vode script.py  # 默认static + stage4

# 旧版
vode --legacy script.py
```

### 方案3：简化为核心参数

保留核心参数：

- `script` (必需)
- `--depth` (默认1)
- `--format` (默认svg)
- `--output` (可选)
- `--legacy` (使用旧版管道)

移除或合并：

- `--stage4` → 成为默认行为
- `--mode` → 仅在--legacy模式下有效
- `--collapse-loops` → 仅在--legacy模式下有效
- `--model-name` → 保留但不常用

## 推荐的最终CLI设计

```bash
# 基本用法（使用所有默认值）
vode script.py

# 指定深度
vode --depth 2 script.py

# 指定输出格式
vode --format png script.py

# 指定输出路径
vode --output result.svg script.py

# 组合参数
vode --depth 2 --format pdf --output model.pdf script.py

# 使用旧版管道（向后兼容）
vode --legacy --depth 5 script.py

# 传递脚本参数
vode script.py --arg1 value1 --arg2 value2
```

这样CLI更简洁，更符合直觉，同时保持向后兼容性。
