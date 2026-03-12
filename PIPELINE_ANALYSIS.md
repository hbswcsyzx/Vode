# VODE 旧版管道与新版管道分析

## 实现机制对比

### 旧版管道（ComputationGraph-based）

#### 静态捕获 (StaticCapture)

**实现方式：**

- 使用 `model.named_modules()` 遍历模块层次结构
- 不运行forward pass
- 创建 `ModuleNode` 表示每个模块
- 通过模块名称（如 "layer1.conv1"）建立父子关系
- 检测 `Sequential` 和 `ModuleList` 作为循环模式

**数据结构：**

```python
class ModuleNode(Node):
    - node_id: str
    - name: str
    - module_type: str
    - params: dict
    - input_shapes: list  # 空（静态模式）
    - output_shapes: list  # 空（静态模式）
```

**特点：**

- 基于模块层次结构
- 创建 `LoopNode` 表示循环模式
- 使用 `ComputationGraph` 存储所有节点和边

#### 动态捕获 (DynamicCapture)

**实现方式：**

- 使用 `register_forward_pre_hook` 和 `register_forward_hook`
- 在forward pass前后捕获输入输出tensor
- 创建 `TensorNode` 和 `ModuleNode`
- 通过 `id(tensor)` 和 `id(module)` 跟踪对象
- 检测模块重用（同一模块被调用多次）

**关键机制：**

```python
# 注册hooks
def _register_hooks(self):
    for module in self.model.modules():
        pre_hook = module.register_forward_pre_hook(self._pre_forward_hook)
        post_hook = module.register_forward_hook(self._post_forward_hook)
        
# 在hook中捕获tensor信息
def _pre_forward_hook(self, module, inputs):
    input_tensors = _flatten_tensors(inputs)
    # 创建TensorNode
    
def _post_forward_hook(self, module, inputs, outputs):
    output_tensors = _flatten_tensors(outputs)
    # 创建TensorNode并建立边
```

**特点：**

- 使用PyTorch的hook机制，**不创建nn子类**
- 捕获实际运行时的tensor信息
- 检测模块重用和循环调用
- 创建显式的数据流边（tensor -> module -> tensor）

### 新版管道（ExecutionNode-based，Stage 4）

#### 静态捕获 (capture_static_execution_graph)

**实现方式：**

- 递归遍历模块层次结构
- 创建 `ExecutionNode` 树形结构
- 每个节点包含 `inputs`, `operation`, `outputs`
- 支持 `is_expandable` 和 `is_expanded` 标志

**数据结构：**

```python
class ExecutionNode:
    - node_id: str
    - name: str
    - depth: int
    - inputs: list[TensorInfo]  # 空（静态模式）
    - operation: OperationInfo
    - outputs: list[TensorInfo]  # 空（静态模式）
    - children: list[ExecutionNode]
    - is_expandable: bool
    - is_expanded: bool
    - parent: ExecutionNode | None
```

**特点：**

- 树形结构，每个节点直接包含子节点
- 不需要单独的Graph对象
- 支持递归深度展开

#### 动态捕获 (DynamicExecutionCapture)

**实现方式：**

- 同样使用 `register_forward_pre_hook` 和 `register_forward_hook`
- 先构建模块层次结构（`_build_module_hierarchy`）
- 在forward pass中填充tensor信息
- 创建 `ExecutionNode` 树形结构

**关键机制：**

```python
def _build_module_hierarchy(self):
    # 预先创建所有ExecutionNode
    for name, module in self.model.named_modules():
        node = ExecutionNode(...)
        self._module_to_node[id(module)] = node
    
    # 建立父子关系
    for name, module in self.model.named_modules():
        # 找到父节点并添加子节点
        parent_node.add_child(node)

# 在hook中填充tensor信息
def _pre_forward_hook(self, module, inputs):
    node = self._module_to_node[id(module)]
    node.inputs = [_tensor_to_tensor_info(t, f"input_{i}") for i, t in enumerate(tensors)]
```

**特点：**

- 同样使用hook机制，**不创建nn子类**
- 预先构建层次结构，运行时填充数据
- 使用 `TensorInfo` 而不是 `TensorNode`
- 树形结构更清晰

## 关键发现

### 1. 都不创建nn子类

**旧版和新版都使用PyTorch的hook机制：**

- `register_forward_pre_hook()` - 在forward前调用
- `register_forward_hook()` - 在forward后调用
- 不需要修改或继承nn.Module

### 2. 核心差异

| 特性 | 旧版 | 新版 |
|------|------|------|
| 数据结构 | 扁平的Graph + 节点列表 | 树形的ExecutionNode |
| 节点类型 | TensorNode, ModuleNode分离 | ExecutionNode统一（包含inputs/operation/outputs） |
| 层次关系 | 通过node_id引用 | 直接的parent/children引用 |
| 循环检测 | 显式的LoopNode | 通过is_expandable标志 |
| 渲染方式 | 节点-边图 | 三列布局（INPUT\|OP\|OUTPUT） |

### 3. 能否融合？

**答案：可以融合，但需要重构**

#### 方案1：统一到ExecutionNode（推荐）

```python
# 保留新版的ExecutionNode作为核心
# 旧版功能通过转换实现

def convert_computation_graph_to_execution_node(graph: ComputationGraph) -> ExecutionNode:
    """将旧版ComputationGraph转换为新版ExecutionNode"""
    # 遍历graph的节点
    # 重建ExecutionNode树
    pass

# 这样旧版API可以继续工作
def capture_static(model):
    # 内部使用新版
    exec_node = capture_static_execution_graph(model)
    # 转换为ComputationGraph（如果需要）
    return convert_to_computation_graph(exec_node)
```

#### 方案2：双模式渲染器

```python
class UnifiedRenderer:
    def render(self, data):
        if isinstance(data, ComputationGraph):
            return self._render_old_style(data)
        elif isinstance(data, ExecutionNode):
            return self._render_new_style(data)
```

### 4. 动态捕获的融合

**旧版和新版动态捕获的核心机制相同：**

- 都使用forward hooks
- 都捕获tensor信息
- 都跟踪模块调用

**差异在于数据组织：**

- 旧版：创建独立的TensorNode和ModuleNode，通过边连接
- 新版：将tensor信息直接嵌入ExecutionNode的inputs/outputs

**融合策略：**

```python
class UnifiedDynamicCapture:
    def capture(self, model, *args, output_format='execution_node'):
        # 使用相同的hook机制
        self._register_hooks()
        
        # 运行forward pass
        model(*args)
        
        # 根据output_format返回不同格式
        if output_format == 'execution_node':
            return self._build_execution_node_tree()
        elif output_format == 'computation_graph':
            return self._build_computation_graph()
```

## 推荐的融合方案

### 阶段1：保持向后兼容

```python
# 旧版API继续工作，内部使用新版实现
def capture_static(model) -> ComputationGraph:
    exec_node = capture_static_execution_graph(model)
    return _convert_to_computation_graph(exec_node)

def capture_dynamic(model, *args) -> ComputationGraph:
    exec_node = capture_dynamic_execution_graph(model, args)
    return _convert_to_computation_graph(exec_node)
```

### 阶段2：统一渲染器

```python
class GraphvizRenderer:
    def render(self, data, style='auto'):
        if style == 'auto':
            style = 'new' if isinstance(data, ExecutionNode) else 'old'
        
        if style == 'new':
            return self.render_execution_graph(data)
        else:
            return self.render_computation_graph(data)
```

### 阶段3：简化CLI

```python
# 移除--stage4标志
# 默认使用新版，--legacy使用旧版

vode script.py                    # 使用ExecutionNode（新版）
vode --legacy script.py           # 使用ComputationGraph（旧版）
vode --legacy --mode dynamic ...  # 旧版动态捕获
```

## 结论

1. **旧版和新版都不创建nn子类**，都使用PyTorch的hook机制
2. **核心差异在数据结构**：扁平图 vs 树形结构
3. **可以融合**：通过转换层保持向后兼容
4. **动态捕获机制相同**：都使用hooks，差异在数据组织
5. **推荐策略**：
   - 新版作为默认（更清晰、更高效）
   - 保留旧版API（向后兼容）
   - 提供转换函数（互操作性）
   - 简化CLI（移除--stage4标志）

## 实施建议

1. **立即可做：**
   - 移除CLI的--stage4标志
   - 将ExecutionNode设为默认
   - 添加--legacy标志用于旧版

2. **短期：**
   - 实现ComputationGraph到ExecutionNode的转换
   - 统一渲染器支持两种格式

3. **长期：**
   - 逐步废弃ComputationGraph
   - 完全迁移到ExecutionNode
   - 移除旧版代码
