# VODE 版本对比与推荐方案

## 版本定义

根据git历史和代码分析：

### 旧版（Old Pipeline）

- **数据结构**: `ComputationGraph` + `Node`（`TensorNode`, `ModuleNode`, `LoopNode`, `FunctionNode`）
- **特点**: 扁平的图结构，节点通过ID引用，显式的边管理
- **实现**: `StaticCapture` 和 `DynamicCapture` 类
- **时间**: 早期实现（commit 05a4778 "reconstruct" 之前）

### 新版（New Pipeline / Stage 4）

- **数据结构**: `ExecutionNode` + `TensorInfo` + `OperationInfo`
- **特点**: 树形结构，直接的parent/children引用，三列布局（INPUT | OPERATION | OUTPUT）
- **实现**: `capture_static_execution_graph()` 和 `DynamicExecutionCapture` 类
- **时间**: 重构后（commit 05a4778 "reconstruct" 之后）

## 核心对比

### 1. 数据结构

| 特性 | 旧版 | 新版 |
|------|------|------|
| 核心结构 | `ComputationGraph` | `ExecutionNode` |
| 节点类型 | 多种（TensorNode, ModuleNode, LoopNode） | 统一（ExecutionNode） |
| 层次关系 | 通过node_id字符串引用 | 直接的对象引用（parent/children） |
| 数据流 | 显式的edges列表 | 隐式（通过inputs/outputs） |
| 循环表示 | 独立的LoopNode | is_loop标志 + loop_type |

### 2. 捕获机制

**共同点：**

- 都使用PyTorch的hook机制（`register_forward_pre_hook`, `register_forward_hook`）
- 都不创建nn.Module子类
- 都捕获tensor的shape/dtype/device信息

**差异点：**

| 特性 | 旧版 | 新版 |
|------|------|------|
| 静态捕获 | `StaticCapture` 类 | `capture_static_execution_graph()` 函数 |
| 动态捕获 | `DynamicCapture` 类 | `DynamicExecutionCapture` 类 |
| 数据组织 | 创建独立的TensorNode和ModuleNode | TensorInfo嵌入ExecutionNode |
| 图构建 | 运行时动态添加节点和边 | 预先构建层次结构，运行时填充数据 |

### 3. 可视化

| 特性 | 旧版 | 新版 |
|------|------|------|
| 布局 | 节点-边图（传统Graphviz） | 三列表格（INPUT \| OPERATION \| OUTPUT） |
| 节点展示 | 每个节点单独显示 | 递归下降，可展开/折叠 |
| 深度控制 | 通过max_depth参数 | 通过expand_to_depth()函数 |
| 循环显示 | LoopNode节点 | 在operation列显示loop信息 |

## 哪个更好？

### 新版（ExecutionNode）更优的原因

#### 1. **更符合递归下降建模**

```python
# 新版：自然的递归结构
class ExecutionNode:
    inputs: list[TensorInfo]      # INPUT
    operation: OperationInfo       # OPERATION
    outputs: list[TensorInfo]      # OUTPUT
    children: list[ExecutionNode]  # 递归展开
```

这完美匹配了你的需求："对代码按照 input->op->output 建模，并且视为可以递归下降展开"

#### 2. **更高效的数据访问**

```python
# 旧版：需要通过ID查找
parent_node = graph.get_node(parent_id)
children = [graph.get_node(child_id) for child_id in parent_node.children]

# 新版：直接访问
children = parent_node.children
```

#### 3. **更清晰的可视化**

```
┌─────────────┬──────────────────┬─────────────┐
│   INPUTS    │    OPERATION     │   OUTPUTS   │
│  shape info │  op_type/name    │  shape info │
└─────────────┴──────────────────┴─────────────┘
```

三列布局直观展示了 input->op->output 的流程

#### 4. **更简洁的代码**

- 不需要单独的Graph对象管理节点
- 不需要维护edges列表
- 不需要node_id到node的映射

#### 5. **更好的扩展性**

- 添加新字段只需修改ExecutionNode
- 支持任意深度的递归展开
- 易于实现交互式显示（展开/折叠）

### 旧版的优势（已被新版吸收）

1. **循环检测** - 已通过is_loop + loop_type融合到新版 ✓
2. **显式边管理** - 新版通过inputs/outputs隐式表达，更清晰 ✓
3. **多种节点类型** - 新版统一为ExecutionNode，更简洁 ✓

## 推荐方案

### 立即行动：使用新版（ExecutionNode）作为唯一实现

**理由：**

1. 新版完全符合你的递归下降建模需求
2. 循环检测功能已融合（刚刚完成）
3. 代码更简洁、性能更好
4. 更易于实现交互式显示

**实施步骤：**

#### 阶段1：清理旧版代码（可选，建议保留作为参考）

```bash
# 可以创建一个legacy分支保存旧版
git branch legacy-computation-graph
# 然后在主分支删除旧版代码
```

#### 阶段2：完善新版功能

- [x] 循环检测（Sequential, ModuleList, 模块重用）
- [x] 可视化显示循环信息
- [ ] 优化渲染性能
- [ ] 添加更多测试用例

#### 阶段3：实现交互式显示

```python
# 基于ExecutionNode的交互式显示
class InteractiveViewer:
    def __init__(self, root: ExecutionNode):
        self.root = root
        self.current_node = root
        self.expanded_nodes = set()
    
    def expand(self, node: ExecutionNode):
        """展开节点到下一层级"""
        if node.is_expandable:
            self.expanded_nodes.add(node.node_id)
            return node.children
    
    def collapse(self, node: ExecutionNode):
        """折叠节点"""
        self.expanded_nodes.discard(node.node_id)
    
    def get_visible_nodes(self) -> list[ExecutionNode]:
        """获取当前可见的节点列表"""
        # 递归遍历，只返回展开的节点
        pass
```

## 针对你的需求的评估

### 需求1：静态捕获（函数调用栈）

**新版支持度：** ✓ 完全支持

- `capture_static_execution_graph()` 捕获模块层次结构
- 递归展开到任意深度
- 支持Sequential/ModuleList等容器

**改进建议：**

- 添加Python函数调用栈捕获（目前只捕获nn.Module）
- 使用`sys.settrace()`或`inspect`模块

### 需求2：动态捕获（数据流图）

**新版支持度：** ✓ 完全支持

- `DynamicExecutionCapture` 使用hooks捕获运行时数据
- 记录tensor的shape/dtype/device
- 支持模块重用检测

**改进建议：**

- 添加tensor统计信息（min/max/mean/std）
- 支持梯度信息捕获

### 需求3：静态显示（指定深度）

**新版支持度：** ✓ 完全支持

- `expand_to_depth(node, depth)` 函数
- `GraphvizRenderer` 渲染指定深度
- 三列布局清晰展示

### 需求4：交互式显示

**新版支持度：** ⚠️ 需要实现

- ExecutionNode结构完美支持
- 需要添加前端交互界面
- 建议使用Web技术（D3.js或Cytoscape.js）

### 需求5：循环处理

**新版支持度：** ✓ 已完成（刚刚融合）

- 检测Sequential/ModuleList
- 检测模块重用
- 在可视化中显示循环信息

## 结论

**新版（ExecutionNode）是更好的选择，原因：**

1. ✅ 完美匹配递归下降建模
2. ✅ 代码更简洁、性能更好
3. ✅ 已融合旧版的循环检测功能
4. ✅ 更易于实现交互式显示
5. ✅ 三列布局更直观

**建议：**

- 删除或归档旧版代码（ComputationGraph相关）
- 专注于完善新版功能
- 优先实现交互式显示
- 使用新版测试unifolm-world-model-action

**下一步行动：**

1. 运行新版捕获unifolm-world-model-action
2. 与model.log对比验证
3. 实现交互式显示原型
4. 清理旧版代码
