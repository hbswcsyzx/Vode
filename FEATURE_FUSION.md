# VODE 旧版独有功能分析与融合方案

## 旧版独有功能清单

### 1. 循环检测（Loop Detection）

#### 功能描述

旧版通过 `LoopNode` 显式检测和表示循环模式：

**检测的循环类型：**

1. **Sequential循环** - 检测 `nn.Sequential` 容器
2. **ModuleList循环** - 检测 `nn.ModuleList` 容器  
3. **模块重用循环** - 检测同一模块被多次调用（动态捕获）

**实现机制：**

```python
# 静态捕获
def _detect_loops(self):
    for name, module in self.model.named_modules():
        if isinstance(module, nn.Sequential):
            self._create_loop_node(node_id, module, "sequential")
        elif isinstance(module, nn.ModuleList):
            self._create_loop_node(node_id, module, "modulelist")

# 动态捕获
def _detect_loops(self):
    for module_id, call_count in self._module_call_count.items():
        if call_count > 1:  # 模块被调用多次
            self._create_loop_node(...)
```

**LoopNode数据结构：**

```python
class LoopNode(Node):
    loop_type: Literal["for", "while", "recursive"]
    iteration_count: int | None
    body_node_ids: list[str]
    is_collapsed: bool
    recursive_call_id: str | None
```

**可视化效果：**

- 折叠模式：显示为单个循环节点（如 "Sequential loop, 3 iterations"）
- 展开模式：显示所有子节点

### 2. 循环折叠控制（collapse_loops）

#### 功能描述

CLI参数 `--collapse-loops` / `--no-collapse-loops` 控制循环节点的显示方式

**使用场景：**

- 折叠：简化视图，适合高层概览
- 展开：查看循环内部细节

### 3. 显式的图结构（ComputationGraph）

#### 功能描述

旧版使用独立的 `ComputationGraph` 对象管理所有节点和边：

```python
class ComputationGraph:
    nodes: dict[str, Node]
    edges: list[tuple[str, str]]
    root_node_ids: list[str]
    node_hierarchy: dict[str, list[str]]
    detected_loops: list[LoopNode]
```

**提供的功能：**

- `get_node(node_id)` - 通过ID获取节点
- `get_children(node_id)` - 获取子节点
- `get_descendants(node_id)` - 获取所有后代
- `add_edge(source, target)` - 添加数据流边
- 统计信息（节点数、边数、最大深度等）

## 新版现有功能

### ExecutionNode树形结构

```python
class ExecutionNode:
    node_id: str
    name: str
    depth: int
    inputs: list[TensorInfo]
    operation: OperationInfo
    outputs: list[TensorInfo]
    children: list[ExecutionNode]
    is_expandable: bool
    is_expanded: bool
    parent: ExecutionNode | None
```

**特点：**

- 直接的父子引用（不需要通过ID查找）
- 内置的展开/折叠标志（`is_expandable`, `is_expanded`）
- 三列布局（INPUT | OPERATION | OUTPUT）

## 功能对比矩阵

| 功能 | 旧版 | 新版 | 是否可融合 |
|------|------|------|-----------|
| 静态捕获 | ✅ | ✅ | - |
| 动态捕获 | ✅ | ✅ | - |
| 深度控制 | ✅ | ✅ | - |
| Sequential检测 | ✅ (LoopNode) | ✅ (is_expandable) | **可融合** |
| ModuleList检测 | ✅ (LoopNode) | ✅ (is_expandable) | **可融合** |
| 模块重用检测 | ✅ (LoopNode) | ❌ | **需添加** |
| 循环折叠控制 | ✅ (collapse_loops) | ✅ (is_expanded) | **已融合** |
| 显式边管理 | ✅ (edges列表) | ❌ (隐式) | **不需要** |
| 节点查找 | ✅ (get_node) | ❌ | **不需要** |
| 统计信息 | ✅ (summary) | ❌ | **可添加** |

## 融合方案

### 方案1：在ExecutionNode中添加循环信息（推荐）

#### 1.1 扩展OperationInfo

```python
@dataclass
class OperationInfo:
    op_type: str
    op_name: str
    params_count: int = 0
    is_composite: bool = False
    
    # 新增：循环信息
    is_loop: bool = False
    loop_type: str | None = None  # "sequential", "modulelist", "reuse"
    iteration_count: int | None = None
```

#### 1.2 在捕获时检测循环

```python
def _module_to_operation_info(module, name):
    op_info = OperationInfo(...)
    
    # 检测Sequential
    if isinstance(module, nn.Sequential):
        op_info.is_loop = True
        op_info.loop_type = "sequential"
        op_info.iteration_count = len(module)
    
    # 检测ModuleList
    elif isinstance(module, nn.ModuleList):
        op_info.is_loop = True
        op_info.loop_type = "modulelist"
        op_info.iteration_count = len(module)
    
    return op_info
```

#### 1.3 动态捕获中检测模块重用

```python
class DynamicExecutionCapture:
    def __init__(self, model):
        self._module_call_count = defaultdict(int)
    
    def _pre_forward_hook(self, module, inputs):
        module_id = id(module)
        self._module_call_count[module_id] += 1
        
        # 如果模块被多次调用，标记为循环
        if self._module_call_count[module_id] > 1:
            node = self._module_to_node[module_id]
            node.operation.is_loop = True
            node.operation.loop_type = "reuse"
            node.operation.iteration_count = self._module_call_count[module_id]
```

#### 1.4 渲染时处理循环

```python
def _format_operation_for_column(self, operation: OperationInfo) -> str:
    parts = [f"<B>{operation.op_type}</B>"]
    
    if operation.op_name and operation.op_name != operation.op_type:
        parts.append(operation.op_name)
    
    if operation.params_count > 0:
        parts.append(f"{self._format_number(operation.params_count)} params")
    
    # 新增：显示循环信息
    if operation.is_loop:
        loop_info = f"{operation.loop_type} loop"
        if operation.iteration_count:
            loop_info += f" ({operation.iteration_count}x)"
        parts.append(loop_info)
    
    if operation.is_composite:
        parts.append("(composite)")
    
    return "<BR/>".join(parts)
```

### 方案2：添加统计信息方法

```python
class ExecutionNode:
    def get_statistics(self) -> dict:
        """获取节点树的统计信息"""
        def count_nodes(node):
            count = 1
            for child in node.children:
                count += count_nodes(child)
            return count
        
        def get_max_depth(node):
            if not node.children:
                return node.depth
            return max(get_max_depth(child) for child in node.children)
        
        def count_loops(node):
            count = 1 if node.operation.is_loop else 0
            for child in node.children:
                count += count_loops(child)
            return count
        
        return {
            "total_nodes": count_nodes(self),
            "max_depth": get_max_depth(self),
            "total_params": self._count_params(self),
            "loop_count": count_loops(self),
        }
```

## 实施计划

### 阶段1：添加循环检测（立即可做）

1. 扩展 `OperationInfo` 添加循环字段
2. 在静态捕获中检测 Sequential/ModuleList
3. 在动态捕获中检测模块重用
4. 更新渲染器显示循环信息

**代码改动：**

- `core/nodes.py` - 扩展 OperationInfo
- `capture/static_capture.py` - 添加循环检测
- `capture/dynamic_capture.py` - 添加重用检测
- `visualize/graphviz_renderer.py` - 显示循环信息

### 阶段2：添加统计方法（可选）

1. 在 ExecutionNode 添加 `get_statistics()` 方法
2. 提供与旧版 ComputationGraph 相同的统计信息

### 阶段3：移除旧版代码

1. 标记旧版API为deprecated
2. 更新文档和示例
3. 在下一个主版本中移除

## 结论

**旧版独有的核心功能只有一个：显式的循环检测和LoopNode**

**融合策略：**

1. ✅ **Sequential/ModuleList检测** - 已通过 `is_expandable` 隐式支持，可显式标记
2. ✅ **模块重用检测** - 需要添加，但实现简单
3. ✅ **循环折叠** - 已通过 `is_expanded` 支持
4. ❌ **显式图结构** - 不需要，树形结构更清晰
5. ❌ **边管理** - 不需要，通过parent/children隐式表达

**推荐行动：**

1. 在 `OperationInfo` 添加 `is_loop`, `loop_type`, `iteration_count` 字段
2. 在捕获时检测并标记循环
3. 在渲染时显示循环信息
4. 完成后可以安全移除旧版代码

**预期效果：**

- 新版功能完全覆盖旧版
- 代码更简洁（树形结构 vs 图结构）
- 性能更好（直接引用 vs ID查找）
- 可视化更清晰（三列布局 + 循环标记）
