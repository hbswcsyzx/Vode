# Vode 技术路线方案

## 技术栈选型

### 后端（Python）

| 组件           | 技术选择                                   | 理由                                  |
|----------------|--------------------------------------------|---------------------------------------|
| **函数流捕获** | `sys.settrace()`                           | Python 内置，无需依赖，捕获所有函数调用 |
| **计算流捕获** | `torch.nn.Module` hooks + `RecorderTensor` | 参考 torchview，成熟方案               |
| **数据序列化** | JSON                                       | 人类可读，易调试，广泛支持              |
| **静态渲染**   | Graphviz                                   | 行业标准，高质量布局                   |
| **Web 服务器** | FastAPI                                    | 快速，异步，自动生成 API 文档           |
| **CLI 框架**   | Click                                      | 简单，功能完整                         |
| **依赖管理**   | Poetry                                     | 现代化，锁定依赖版本                   |

### 前端（交互式查看器）

| 组件         | 技术选择   | 理由                         |
|--------------|------------|------------------------------|
| **框架**     | React 18   | 组件化，生态丰富              |
| **图形库**   | React Flow | 专为节点图设计，支持拖拽、缩放 |
| **UI 组件**  | Ant Design | 专业，组件丰富，中文友好       |
| **状态管理** | Zustand    | 轻量，简单                    |
| **构建工具** | Vite       | 快速，现代化                  |
| **类型检查** | TypeScript | 类型安全                     |

### 未来扩展

| 组件            | 技术选择                   | 理由                |
|-----------------|----------------------------|---------------------|
| **VSCode 扩展** | VSCode Extension API       | 原生集成            |
| **可视化编辑**  | React Flow + Monaco Editor | 图形编辑 + 代码编辑 |

---

## 核心模块

### 1. 捕获模块 (`src/vode/capture/`)

```
capture/
├── function_tracer.py      # 函数流捕获（sys.settrace）
├── computation_tracer.py   # 计算流捕获（hooks + RecorderTensor）
├── recorder_tensor.py      # RecorderTensor 实现
└── utils.py                # 数据序列化工具
```

**核心逻辑**：

```python
# function_tracer.py
class FunctionTracer:
    def start(): sys.settrace(callback)
    def stop(): sys.settrace(None)
    def to_graph(): return {...}

# computation_tracer.py
class ComputationTracer:
    def trace(model, input): 
        register_hooks()
        output = model(input)
        return output
    def to_graph(): return {...}
```

### 2. 核心数据结构 (`src/vode/core/`)

```
core/
├── graph.py       # Graph, Node, Edge 数据类
├── serializer.py  # JSON 序列化/反序列化
└── validator.py   # 数据验证
```

**核心逻辑**：

```python
# graph.py
@dataclass
class Node:
    id: str
    type: str
    name: str
    depth: int
    children: List[str]
    data: Dict[str, Any]

@dataclass
class Graph:
    metadata: Dict
    nodes: List[Node]
    edges: List[Edge]
    
    def to_json() -> str
    def from_json(json_str) -> Graph
```

### 3. 可视化模块 (`src/vode/visualize/`)

```
visualize/
├── graphviz_renderer.py   # 静态图片导出
├── server.py              # FastAPI 服务器
└── static/                # 前端构建产物
    └── index.html
```

**核心逻辑**：

```python
# graphviz_renderer.py
class GraphvizRenderer:
    def render(graph, depth, output_file):
        dot = build_graphviz(graph, depth)
        dot.render(output_file)

# server.py
@app.get("/api/graph")
def get_graph(file): return load_json(file)

@app.get("/api/node/{id}")
def get_node_details(id): return node_data
```

### 4. CLI 模块 (`src/vode/cli.py`)

```python
@click.group()
def cli(): pass

@cli.command()
def trace(script, mode, capture_data, output):
    if mode == 'function':
        tracer = FunctionTracer(capture_data)
        tracer.start()
        exec_script(script)
        tracer.stop()
    else:
        tracer = ComputationTracer(capture_data)
        # 需要特殊处理：解析脚本找到模型和输入
    
    save_json(tracer.to_graph(), output)

@cli.command()
def export(trace_file, output, mode, depth):
    graph = load_json(trace_file)
    renderer = GraphvizRenderer(graph, mode, depth)
    renderer.render(output)

@cli.command()
def view(trace_file, port):
    start_server(trace_file, port)
    open_browser(f"http://localhost:{port}")
```

### 5. 前端模块 (`frontend/`)

```
frontend/
├── src/
│   ├── components/
│   │   ├── GraphView.tsx          # React Flow 主视图
│   │   ├── NodeDetailsPanel.tsx   # 节点详情面板
│   │   ├── Toolbar.tsx            # 搜索、过滤、控制
│   │   ├── CustomNode.tsx         # 自定义节点渲染
│   │   └── MiniMap.tsx            # 小地图
│   ├── hooks/
│   │   ├── useGraph.ts            # 加载图数据
│   │   ├── useNodeExpansion.ts   # 展开/折叠逻辑
│   │   └── useSearch.ts           # 搜索功能
│   ├── types/
│   │   └── graph.ts               # TypeScript 类型定义
│   ├── App.tsx
│   └── main.tsx
├── package.json
└── vite.config.ts
```

**核心逻辑**：

```typescript
// GraphView.tsx
function GraphView({ traceFile }) {
  const { graph } = useGraph(traceFile);
  const { expandNode, collapseNode } = useNodeExpansion();
  
  return (
    <ReactFlow
      nodes={visibleNodes}
      edges={visibleEdges}
      onNodeClick={handleNodeClick}
    />
  );
}

// useNodeExpansion.ts
function expandNode(nodeId) {
  // 从 graph 中获取子节点
  // 添加到 visibleNodes
  // 更新布局
}
```

---

## 实施阶段

### 阶段 1：基础捕获（4周）

**目标**：实现两种捕获机制，生成 JSON 文件。

**任务**：

1. 实现 `FunctionTracer`（`sys.settrace` 方案）
2. 实现 `ComputationTracer`（参考 torchview）
3. 定义 JSON 数据格式
4. 实现 `vode trace` 命令
5. 编写单元测试

**交付物**：

- 可运行的 `vode trace` 命令
- 生成的 `.vode.json` 文件
- 测试覆盖率 > 80%

---

### 阶段 2：静态导出（2周）

**目标**：使用 Graphviz 导出静态图片。

**任务**：

1. 实现 `GraphvizRenderer`
2. 支持深度过滤
3. 支持多种输出格式（PNG, SVG, PDF）
4. 实现 `vode export` 命令
5. 优化图形布局和样式

**交付物**：

- 可运行的 `vode export` 命令
- 高质量的可视化图片
- 示例图片库

---

### 阶段 3：交互式查看器（6周）

**目标**：构建 React Flow 交互式查看器。

**任务**：

1. 搭建前端项目（React + Vite + TypeScript）
2. 实现 FastAPI 后端服务器
3. 实现 React Flow 图形展示
4. 实现节点展开/折叠逻辑
5. 实现节点详情面板
6. 实现搜索和过滤功能
7. 实现 `vode view` 命令
8. 优化性能（虚拟化、懒加载）

**交付物**：

- 可运行的 `vode view` 命令
- 功能完整的交互式查看器
- 用户文档

---

### 阶段 4：数据捕获（2周）

**目标**：完善数据捕获功能。

**任务**：

1. 实现函数参数捕获
2. 实现张量统计信息捕获
3. 优化数据序列化（避免文件过大）
4. 在详情面板中展示数据
5. 实现数据导出功能

**交付物**：

- 完整的数据捕获功能
- 详情面板展示所有数据
- 性能优化（文件大小 < 10MB）

---

### 阶段 5：可视化编辑器（8周，未来）

**目标**：实现 Scratch 风格的可视化编程界面。

**任务**：

1. 设计模块库（左侧面板）
2. 实现拖拽功能
3. 实现连线功能
4. 实现属性配置面板
5. 实现代码生成
6. 实现双向同步（代码 ↔ 图形）
7. 实现模板系统
8. 实现 `vode editor` 命令

**交付物**：

- 可运行的 `vode editor` 命令
- 完整的可视化编程界面
- 代码生成器
- 模板库

---

## 关键技术挑战与解决方案

### 挑战 1：性能开销

**问题**：`sys.settrace()` 会导致 10-20x 的性能下降。

**解决方案**：

- 提供采样模式（每 N 次调用记录一次）
- 过滤标准库和第三方库
- 限制最大深度
- 提供 `--fast` 模式（只记录关键信息）

### 挑战 2：大图渲染

**问题**：大型程序可能产生 10,000+ 节点的图。

**解决方案**：

- 虚拟化渲染（只渲染可见节点）
- 懒加载（按需加载子节点）
- 聚合显示（合并重复调用）
- 分页显示

### 挑战 3：张量数据过大

**问题**：捕获所有张量数据会导致文件过大。

**解决方案**：

- 只存储统计信息（min, max, mean, std）
- 只存储前 N 个值（如前 10 个）
- 提供 `--no-capture-data` 选项
- 使用压缩（gzip）

### 挑战 4：计算流捕获的准确性

**问题**：如何准确追踪张量在模块间的流动？

**解决方案**：

- 参考 torchview 的 `RecorderTensor` 方案
- 重写 `__torch_function__` 拦截操作
- 使用 context stack 追踪层级关系
- 处理 inplace 操作的特殊情况

---

## 开发优先级

### P0（必须）

- [x] 函数流捕获
- [x] 计算流捕获
- [ ] JSON 序列化
- [ ] `vode trace` 命令

### P1（重要）

- [ ] Graphviz 静态导出
- [ ] `vode export` 命令
- [ ] 基础测试

### P2（需要）

- [ ] FastAPI 服务器
- [ ] React Flow 前端
- [ ] 节点展开/折叠
- [ ] `vode view` 命令

### P3（增强）

- [ ] 数据捕获
- [ ] 详情面板
- [ ] 搜索和过滤
- [ ] 性能优化

### P4（未来）

- [ ] 可视化编辑器
- [ ] VSCode 扩展
- [ ] 模板系统
- [ ] 代码生成

---

## 时间规划

| 阶段               | 周数     | 核心交付           |
|--------------------|----------|--------------------|
| **阶段1：基础捕获** | 4周      | `vode trace` 可用  |
| **阶段2：静态导出** | 2周      | `vode export` 可用 |
| **阶段3：交互查看** | 6周      | `vode view` 可用   |
| **阶段4：数据捕获** | 2周      | 完整数据展示       |
| **阶段5：可视编辑** | 8周      | `vode editor` 可用 |
| **总计**           | **22周** | **完整功能**       |

---

## 架构设计原则

### 1. 模块化

- 捕获、存储、展示三层分离
- 每个模块独立测试
- 接口清晰，易于扩展

### 2. 可扩展性

- 支持插件机制（自定义捕获器、渲染器）
- 配置文件驱动（`.voderc`）
- 开放 API（Python API + REST API）

### 3. 性能优先

- 懒加载
- 增量渲染
- 缓存机制
- 异步处理

### 4. 用户友好

- 零配置启动
- 清晰的错误提示
- 丰富的文档和示例
- 渐进式学习曲线

---

## 技术债务管理

### 已知限制

1. **函数流捕获性能开销大**：未来考虑 C 扩展或 Cython 优化
2. **计算流仅支持 PyTorch**：未来扩展到 TensorFlow, JAX
3. **JSON 文件可能过大**：未来支持二进制格式（MessagePack）
4. **不支持分布式追踪**：未来支持多进程/多GPU

### 重构计划

- 阶段1结束后：重构数据结构，确保可扩展
- 阶段3结束后：性能优化，处理大图
- 阶段5开始前：架构重审，确保支持可视化编辑

---

## 测试策略

### 单元测试

- 每个模块独立测试
- 覆盖率 > 80%
- 使用 pytest

### 集成测试

- 端到端测试：trace → export → view
- 真实模型测试：ResNet, BERT, GPT

### 性能测试

- 大图测试（10,000+ 节点）
- 深递归测试（100+ 层）
- 内存泄漏检测

---

## 部署方案

### PyPI 发布

```bash
poetry build
poetry publish
```

### Docker 镜像（可选）

```dockerfile
FROM python:3.10
RUN pip install vode
CMD ["vode", "view"]
```

### VSCode 扩展（未来）

```bash
vsce package
vsce publish
```

---

## 参考资源

- **torchview**: 计算流捕获的参考实现
- **React Flow**: 交互式图形库文档
- **FastAPI**: Web 服务器框架
- **Graphviz**: 静态图形渲染
