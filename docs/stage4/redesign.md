# VODE Stage 4: 可视化重新设计

**日期**: 2026-03-11  
**目的**: 修复当前可视化实现的问题，正确体现 input->operation->output 的递归下降模型

## 当前问题分析

### 问题 1: 节点布局不正确

**现状**: 每个节点显示为单独的表格，没有体现 INPUT-OP-OUTPUT 的水平三列布局

**期望**:

```
┌─────────┬──────────────┬─────────┐
│  INPUT  │   OPERATION  │  OUTPUT │
└─────────┴──────────────┴─────────┘
```

### 问题 2: 深度控制不正确

**现状**: 深度控制只是简单地截断节点，没有正确处理递归展开

**期望**:

- 当 depth=0 时，只显示顶层：`INPUT -> Sequential -> OUTPUT`
- 当 depth=1 时，展开一层：`INPUT -> Linear -> ReLU -> Linear -> OUTPUT`
- 不应该同时显示 Sequential 和其内部的 Linear/ReLU

### 问题 3: 重复节点

**现状**: 同一个操作可能被渲染多次

**期望**: 每个操作只渲染一次，通过深度参数控制展开级别

### 问题 4: 数据流不清晰

**现状**: 中间节点也显示 INPUT/OUTPUT，导致混乱

**期望**: 只有图的左端和右端有数据流节点，中间只有操作节点

## 新设计方案

### 核心思路

1. **递归展开模型**：
   - 每个节点都是 `(inputs, operation, outputs)` 三元组
   - 节点可以递归展开：`operation` 可以展开为子图
   - 深度参数控制展开级别

2. **渲染策略**：
   - 根据深度参数决定哪些节点展开，哪些折叠
   - 折叠的节点显示为单个 operation
   - 展开的节点显示为子图（多个 operation 的序列）

3. **布局规则**：
   - 水平布局：从左到右表示数据流
   - 每个 operation 节点显示为三列表格：INPUT | OP | OUTPUT
   - 节点之间用箭头连接，表示数据流

### 数据结构设计

```python
class ExecutionNode:
    """表示一次执行的节点"""
    node_id: str
    name: str
    depth: int
    
    # 核心三元组
    inputs: List[TensorInfo]      # 输入数据
    operation: OperationInfo       # 操作信息
    outputs: List[TensorInfo]      # 输出数据
    
    # 递归展开
    children: List[ExecutionNode]  # 子节点（展开后的内容）
    is_expandable: bool            # 是否可以展开
    is_expanded: bool              # 当前是否已展开

class TensorInfo:
    """Tensor 信息"""
    name: str
    shape: tuple
    dtype: str
    device: str
    
class OperationInfo:
    """操作信息"""
    op_type: str          # 操作类型（Linear, ReLU, Sequential等）
    op_name: str          # 操作名称
    params_count: int     # 参数数量
    is_composite: bool    # 是否是复合操作（可展开）
```

### 渲染逻辑

```python
def render_graph(root_node, max_depth):
    """渲染图形"""
    # 1. 根据 max_depth 决定展开策略
    expanded_nodes = expand_to_depth(root_node, max_depth)
    
    # 2. 构建线性序列（展开后的操作序列）
    operation_sequence = flatten_to_sequence(expanded_nodes)
    
    # 3. 渲染为 Graphviz
    #    - 左端：INPUT 节点
    #    - 中间：OPERATION 节点序列
    #    - 右端：OUTPUT 节点
    return render_to_graphviz(operation_sequence)

def expand_to_depth(node, max_depth, current_depth=0):
    """递归展开到指定深度"""
    if current_depth >= max_depth or not node.is_expandable:
        # 不展开，返回折叠的节点
        return [node]
    else:
        # 展开，返回子节点
        result = []
        for child in node.children:
            result.extend(expand_to_depth(child, max_depth, current_depth + 1))
        return result
```

### 可视化示例

**depth=0** (折叠):

```
┌─────────┐     ┌──────────────┐     ┌─────────┐
│ INPUT   │ --> │  Sequential  │ --> │ OUTPUT  │
│ (1,10)  │     │  3 layers    │     │ (1,10)  │
└─────────┘     └──────────────┘     └─────────┘
```

**depth=1** (展开一层):

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
│ INPUT   │ --> │  Linear  │ --> │   ReLU   │ --> │  Linear  │ --> │ OUTPUT  │
│ (1,10)  │     │  10->20  │     │          │     │  20->10  │     │ (1,10)  │
└─────────┘     └──────────┘     └──────────┘     └──────────┘     └─────────┘
```

### 循环处理

对于循环结构（如 Sequential, ModuleList），有两种处理方式：

1. **折叠显示**（默认）：

```
┌──────────────────┐
│  Sequential      │
│  [Linear, ReLU]  │
│  x 3 iterations  │
└──────────────────┘
```

1. **展开显示**（当深度足够时）：

```
Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU
```

### 实现计划

1. **修改数据结构** (`src/vode/core/nodes.py`):
   - 确保每个节点都有 `inputs`, `operation`, `outputs`
   - 添加 `is_expandable`, `is_expanded` 标志

2. **修改捕获逻辑** (`src/vode/capture/`):
   - 静态捕获：构建完整的层次结构
   - 动态捕获：记录实际的数据流和 tensor 信息

3. **重写渲染器** (`src/vode/visualize/graphviz_renderer.py`):
   - 实现 `expand_to_depth()` 函数
   - 实现 `flatten_to_sequence()` 函数
   - 修改 Graphviz 生成逻辑，使用水平布局
   - 每个节点渲染为三列表格

4. **测试验证**:
   - 测试不同深度参数的效果
   - 测试循环结构的处理
   - 在 unifolm-world-model-action 上验证

## 关键设计决策

### 决策 1: 节点表示

**选择**: 每个节点都是 `(inputs, operation, outputs)` 三元组

**理由**:

- 符合 input->op->output 的核心建模思路
- 支持递归展开
- 清晰表达数据流

### 决策 2: 深度控制

**选择**: 深度参数控制递归展开级别，而不是简单截断

**理由**:

- 避免重复显示（不会同时显示 Sequential 和其内部操作）
- 用户可以通过调整深度参数控制详细程度
- 符合递归下降的思路

### 决策 3: 布局方式

**选择**: 水平线性布局，从左到右表示数据流

**理由**:

- 直观表达数据流向
- 避免复杂的树形布局
- 易于理解和实现

### 决策 4: 循环处理

**选择**: 默认折叠，可选展开

**理由**:

- 节省空间
- 保持图形简洁
- 用户可以通过深度参数控制

## 下一步行动

1. 创建详细的实现计划文档
2. 修改数据结构
3. 重写渲染器
4. 测试验证
5. 在 unifolm-world-model-action 上验证

## 参考

- Stage 3 设计文档：`/home/zyf/XXX/vode/docs/stage3/`
- 当前实现：`/home/zyf/XXX/vode/src/vode/`
- Torchview 参考：`/home/zyf/XXX/torchview/`
