# VODE 当前状态文档

**日期**: 2026-03-11  
**目的**: 对照原始任务要求，记录当前代码库实现的功能和依赖文件

## 原始任务要求回顾

### 核心目标

打造一个 "view your code" 的工具，名为 vode，实现以下功能：

**两种捕获方式**：

1. **静态捕获**：捕获所有函数的调用栈，从最顶层的 main 函数展开到最底层的算子（如 torch.nn 中的算子）
2. **动态捕获**：只捕获程序在初始化阶段后，最终运行时的数据流动图

**两种显示方式**：

1. **静态显示**：用户指定深入层级，绘制从顶层到该层的内容
2. **交互式显示**（未完成）：允许用户深入或退出到某一层次，查看节点详细信息

**核心建模思路**：

- 使用 `input -> operation -> output` 的递归下降模型
- 函数：input=参数，op=函数本身，output=返回值
- 算子：input=tensor/number，op=算子本身，output=结果
- 支持递归展开（如 Sequential list 可展开为多个 nn 算子）
- 循环部分需要特殊处理，避免完全展开占用空间

**测试要求**：

- 在 unifolm-world-model-action 模型上测试
- 与 model.log 中的 torch 打印结构对比
- 捕获内容只能多不能少

## 当前实现状态

### 已完成的功能（对应原始任务）

#### ✅ 静态捕获 + 静态显示

**位置**: `capture/static_capture.py` + `visualize/graphviz_renderer.py`

**实现内容**：

- 使用 `model.named_modules()` 遍历模块层次结构
- 构建 `ModuleNode` 树，包含父子关系
- 统计参数数量（总数和可训练参数）
- 检测 `nn.Sequential` 和 `nn.ModuleList` 作为循环模式
- 支持深度控制（`max_depth` 参数）
- 输出 Graphviz 格式（.gv, .svg, .png, .pdf）

**核心文件**：

- `capture/static_capture.py` - 静态捕获实现
- `core/nodes.py` - 节点数据结构（Node, ModuleNode, LoopNode等）
- `core/graph.py` - ComputationGraph 容器
- `visualize/graphviz_renderer.py` - Graphviz 渲染器
- `visualize/visualizer.py` - 可视化 API

**测试验证**：

- ✅ 在 unifolm-world-model-action 上测试成功
- ✅ 捕获了 4097 个模块
- ✅ model.log 中的所有模块都已捕获
- ✅ 生成了多个深度的可视化（full, d3, d5）

#### ✅ 动态捕获 + 静态显示

**位置**: `capture/dynamic_capture.py` + `visualize/graphviz_renderer.py`

**实现内容**：

- 使用 PyTorch forward hooks 拦截模块执行
- 捕获实际运行时的 tensor 形状、dtype、device
- 跟踪模块重用（同一模块实例多次调用）
- 构建数据流图，包含 TensorNode 和实际形状信息
- 支持多种输入类型（tensor, tuple, dict）
- 可选计算统计信息（min, max, mean, std）

**核心文件**：

- `capture/dynamic_capture.py` - 动态捕获实现
- `core/nodes.py` - TensorNode, FunctionNode 数据结构
- `core/utils.py` - Tensor 信息提取工具
- `visualize/graphviz_renderer.py` - 同一渲染器支持两种模式

**测试验证**：

- ✅ 基础测试通过
- ✅ 支持嵌套模型和复杂结构
- ✅ 正确捕获 tensor 元数据

### 核心数据结构（符合递归下降思路）

**位置**: `core/nodes.py`

```python
class Node:
    """基础节点：input -> operation -> output"""
    - inputs: List[Node]      # 输入节点
    - operation: str          # 操作名称
    - outputs: List[Node]     # 输出节点
    - children: List[Node]    # 子节点（支持递归展开）
    - depth: int              # 层级深度

class TensorNode(Node):
    """Tensor 数据节点"""
    - shape: tuple            # 形状
    - dtype: str              # 数据类型
    - device: str             # 设备
    - statistics: dict        # 统计信息

class ModuleNode(Node):
    """模块节点（nn.Module）"""
    - module_type: str        # 模块类型
    - parameters_count: int   # 参数数量
    - is_leaf: bool           # 是否叶子模块

class LoopNode(Node):
    """循环节点（Sequential, ModuleList等）"""
    - loop_type: str          # 循环类型
    - iterations: int         # 迭代次数
    - iteration_nodes: List   # 每次迭代的节点
```

### API 使用方法

```python
from vode import capture_static, capture_dynamic, visualize

# 静态捕获 + 静态显示
graph = capture_static(model)
visualize(graph, output_path='model.svg', max_depth=5)

# 动态捕获 + 静态显示
graph = capture_dynamic(model, x)
visualize(graph, output_path='model_dynamic.svg', max_depth=3)

# 或使用便捷函数
from vode import vode
vode(model, mode='static', output='model.svg', max_depth=5)
vode(model, x, mode='dynamic', output='model_dynamic.svg')
```

## 测试验证结果

### Unifolm 模型测试

**测试脚本**: `tests/test_unifolm.py`  
**测试结果**: ✅ 通过

**捕获结果**：

- 模型：LatentVisualDiffusion (WMAModel)
- 捕获模块数：4097 个
- 生成文件：
  - `test_outputs/unifolm_static_full.gv` - 完整结构
  - `test_outputs/unifolm_static_d3.gv` - 深度3
  - `test_outputs/unifolm_static_d5.gv` - 深度5

**与 model.log 对比**：

- ✅ 所有 model.log 中的模块都已捕获
- ✅ 捕获了更多细节（完整的模块路径）
- ✅ 包含参数统计信息

## 依赖文件清单

### 核心实现文件（必须保留）

```text
vode/
├── __init__.py
├── capture/
│   ├── __init__.py
│   ├── static_capture.py
│   └── dynamic_capture.py
├── core/
│   ├── __init__.py
│   ├── nodes.py
│   ├── graph.py
│   └── utils.py
├── visualize/
│   ├── __init__.py
│   ├── graphviz_renderer.py
│   ├── visualizer.py
│   └── vode_wrapper.py
├── pyproject.toml
└── README.md
```

### 示例和测试（保留）

```text
vode/
├── examples/
│   ├── simple_example.py
│   ├── advanced_example.py
│   └── large_model_example.py
├── tests/
│   ├── test_unifolm.py
│   └── test_visualization.py
└── test_outputs/
    └── *.gv, *.svg（生成的文件）
```

### 临时文件（可删除）

```text
vode/
├── test_depth_fix.py
├── test_io_nodes.py
├── test_loop_detection.py
├── test_nested_fix.py
├── test_debug.py
├── debug_edges.py
├── debug_nested.py
├── regenerate_graphs.py
├── complete_nested_dataflow.gv
└── complete_nested_dataflow.d2
```

## 总结

### 任务完成情况

| 任务要求          | 完成状态 | 实现位置                         |
|-------------------|----------|----------------------------------|
| 静态捕获          | ✅ 完成   | `capture/static_capture.py`      |
| 动态捕获          | ✅ 完成   | `capture/dynamic_capture.py`     |
| 静态显示          | ✅ 完成   | `visualize/graphviz_renderer.py` |
| 交互式显示        | ❌ 未要求 | 任务明确说明暂不实现             |
| 递归下降建模      | ✅ 完成   | `core/nodes.py`                  |
| 循环处理          | ✅ 完成   | `LoopNode` + 折叠功能            |
| 深度控制          | ✅ 完成   | `max_depth` 参数                 |
| Unifolm 测试      | ✅ 通过   | `tests/test_unifolm.py`          |
| 与 model.log 对比 | ✅ 通过   | 捕获内容 >= model.log            |

### 核心成果

1. **完整实现了任务要求的两种组合**：
   - 静态捕获 + 静态显示
   - 动态捕获 + 静态显示

2. **符合递归下降建模思路**：
   - input -> operation -> output 模式
   - 支持递归展开
   - 循环节点特殊处理

3. **通过实际测试验证**：
   - 在复杂模型（4097个模块）上测试成功
   - 捕获内容完整，满足要求

4. **提供友好的 API**：
   - Python API：`capture_static()`, `capture_dynamic()`, `visualize()`
   - 便捷函数：`vode()`

### 代码库清理建议

**可以删除的临时文件**：

- 根目录下的 `test_*.py`, `debug_*.py` 等测试脚本
- 生成的 `.gv`, `.d2` 文件

**必须保留的核心文件**：

- `capture/`, `core/`, `visualize/` 目录及其内容
- `**init**.py`, `pyproject.toml`, `README.md`
- `examples/`, `tests/` 目录

**可选保留的扩展功能**：

- `src/vode/` 目录（额外的高级功能，不在原始任务要求中）
- `docs/` 目录（开发文档）
