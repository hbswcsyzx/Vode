# Vode 技术栈与架构设计

基于 README.md 的需求分析，本文档详细说明 Vode 的技术选型、数据采集方案、可视化方案及可参考的现有技术。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────┐
│  CLI 入口层 (vode command)                              │
│  - 命令行参数解析                                        │
│  - 启动 Python 进程并注入追踪代码                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  数据采集层 (Tracer/Collector)                          │
│  - 多层级追踪引擎                                        │
│  - 变量名捕获                                            │
│  - 运行时值记录                                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  数据存储层 (Graph Builder)                             │
│  - 计算图构建                                            │
│  - 层级关系管理                                          │
│  - 数据序列化                                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  可视化层 (Renderer)                                    │
│  - 静态图生成 (Graphviz)                                │
│  - 交互式 Web 界面 (React + 图形库)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 二、技术栈选型

### 2.1 CLI 入口层

**技术选择：Python + Click/Typer**

- **Click** 或 **Typer**：现代化的 CLI 框架，支持子命令、参数验证、帮助文档自动生成
- **实现方式**：

  ```python
  # vode/cli.py
  import click
  import subprocess
  import sys
  
  @click.command()
  @click.argument('script', nargs=-1, required=True)
  @click.option('--output', '-o', help='输出文件路径')
  @click.option('--web', is_flag=True, help='启动 Web 界面')
  @click.option('--depth', type=int, default=-1, help='追踪深度')
  def main(script, output, web, depth):
      # 1. 设置环境变量传递配置
      # 2. 使用 subprocess 启动 Python 并注入追踪代码
      # 3. 等待执行完成，处理输出
      pass
  ```

**参考实现**：

- `mpirun`、`torchrun` 的命令行包装方式
- `pytest` 的插件注入机制

---

### 2.2 数据采集层

这是核心难点，需要在多个层级捕获信息。

#### 2.2.1 PyTorch 层级追踪

**技术选择：PyTorch Hooks + RecorderTensor**

**参考 torchview 的实现**：

- `torchview/src/torchview/recorder_tensor.py`：包装 Tensor，记录操作历史
- `torchview/src/torchview/torchview.py`：使用 `register_forward_hook` 和 `register_forward_pre_hook`

**核心技术**：

```python
# 1. 注册 nn.Module 的 hooks
def register_hooks(model):
    for name, module in model.named_modules():
        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)

# 2. 使用 RecorderTensor 包装输入
class RecorderTensor(torch.Tensor):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # 记录每个 torch 函数调用
        result = super().__torch_function__(func, types, args, kwargs)
        # 记录到计算图
        return result
```

**优点**：

- torchview 已验证可行性
- 可以捕获 `nn.Module` 和 `torch` 函数级别的操作

**局限**：

- 无法捕获纯 Python 函数
- 无法获取变量名

#### 2.2.2 Python 函数级追踪

**技术选择：sys.settrace() 或 sys.monitoring (Python 3.12+)**

**参考 pdb/debugger 的实现**：

```python
import sys
import inspect

def trace_function(frame, event, arg):
    if event == 'call':
        # 函数调用
        func_name = frame.f_code.co_name
        local_vars = frame.f_locals
        # 记录函数调用和参数
    elif event == 'return':
        # 函数返回
        return_value = arg
        # 记录返回值
    return trace_function

sys.settrace(trace_function)
```

**优点**：

- 可以捕获所有 Python 函数调用
- 可以访问局部变量（`frame.f_locals`）

**缺点**：

- 性能开销大（每行代码都会触发）
- 需要过滤掉标准库和框架内部调用

**优化方案**：

- 使用 `sys.monitoring`（Python 3.12+）：性能更好，粒度可控
- 只追踪用户代码，跳过标准库

#### 2.2.3 变量名捕获

**技术选择：inspect 模块 + AST 分析**

**方法 1：运行时反向查找**

```python
import inspect

def get_variable_name(obj):
    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals
    # 反向查找：哪个变量名指向这个对象
    for name, value in local_vars.items():
        if id(value) == id(obj):
            return name
    return None
```

**方法 2：AST 静态分析**

```python
import ast

# 解析源代码，提取变量赋值关系
source = inspect.getsource(func)
tree = ast.parse(source)
# 分析 AST 找到变量名
```

**参考**：

- Python debugger 的变量查看功能
- IPython 的 `%whos` 实现

**挑战**：

- 同一对象可能有多个名称（别名）
- 需要维护变量名在不同作用域的映射

#### 2.2.4 运行时值捕获

**技术选择：直接访问 + 采样策略**

```python
def capture_value(tensor):
    if tensor.numel() > 1000:  # 大张量
        return {
            'type': 'statistics',
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
        }
    else:  # 小张量
        return {
            'type': 'full',
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'value': tensor.detach().cpu().numpy().tolist(),
        }
```

**参考**：

- PyTorch debugger 的 tensor 查看
- TensorBoard 的 histogram 记录

**内存优化**：

- 默认只记录统计信息
- 用户点击时才加载完整值（需要保存 checkpoint）
- 或者在运行时就保存完整值到磁盘

---

### 2.3 数据存储层

#### 2.3.1 数据结构设计

**核心数据结构：递归的计算图**

```python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Variable:
    """变量信息"""
    name: str                    # 变量名
    shape: tuple                 # 形状
    dtype: str                   # 数据类型
    device: str                  # 设备
    var_type: str                # argument/parameter/constant
    value_ref: Optional[str]     # 值的引用（文件路径或 ID）
    statistics: Optional[dict]   # 统计信息

@dataclass
class Operation:
    """操作节点"""
    id: str                      # 唯一标识
    name: str                    # 操作名称（函数名、模块名等）
    type: str                    # 类型（function/module/operator/primitive）
    inputs: Dict[str, Variable]  # 输入字典
    outputs: Dict[str, Variable] # 输出字典
    children: List['Operation']  # 子操作（展开后的内容）
    metadata: dict               # 额外信息（源码位置、执行时间等）

@dataclass
class ComputationGraph:
    """完整的计算图"""
    root: Operation              # 根节点
    variables: Dict[str, Variable]  # 所有变量的索引
    edges: List[tuple]           # 数据流边 (src_var_id, dst_var_id)
```

#### 2.3.2 序列化格式

**选择：JSON + 外部二进制文件**

**方案 1：纯 JSON**

- 优点：人类可读，Web 友好，易于调试
- 缺点：大模型的完整值会导致文件巨大

**方案 2：JSON + HDF5/NPZ**

```
output/
├── graph.json          # 计算图结构和元数据
├── values/
│   ├── var_001.npy     # 变量值（NumPy 格式）
│   ├── var_002.npy
│   └── ...
```

**方案 3：Protocol Buffers**

- 优点：高效、类型安全、支持增量解析
- 缺点：需要编译 .proto 文件，调试不便

**推荐：JSON + NPZ**

- 结构用 JSON（易于 Web 加载）
- 大数组用 NPZ（压缩存储）
- 参考 TensorBoard 的事件文件格式

```python
import json
import numpy as np

# 保存
graph_data = {
    'operations': [...],
    'variables': [...],
    'edges': [...]
}
with open('graph.json', 'w') as f:
    json.dump(graph_data, f)

# 保存变量值
np.savez_compressed('values.npz', 
                    var_001=tensor1.numpy(),
                    var_002=tensor2.numpy())
```

---

### 2.4 可视化层

#### 2.4.1 静态图生成

**技术选择：Graphviz**

**参考 torchview 的实现**：

- `torchview/src/torchview/torchview.py` 使用 `graphviz` Python 库
- 生成 DOT 格式，渲染为 SVG/PNG

```python
from graphviz import Digraph

def render_static(graph: ComputationGraph, output_path: str):
    dot = Digraph(comment='Computation Graph')
    dot.attr(rankdir='TB')  # 从上到下
    
    # 添加节点
    for op in graph.operations:
        label = f"{op.name}\n{op.type}"
        dot.node(op.id, label, shape='box')
    
    # 添加边
    for src, dst in graph.edges:
        dot.edge(src, dst)
    
    dot.render(output_path, format='svg')
```

**优点**：

- 成熟稳定，布局算法优秀
- 支持多种输出格式

**缺点**：

- 不支持交互
- 大图性能差

#### 2.4.2 交互式 Web 界面

这是核心挑战，需要实现：

- 选择性展开/折叠
- 点击查看详情
- 搜索/过滤
- 流畅的性能（大图）

**技术栈选择**：

**后端：FastAPI**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/api/graph")
def get_graph():
    # 返回计算图 JSON
    return graph_data

@app.get("/api/variable/{var_id}")
def get_variable_detail(var_id: str):
    # 返回变量详细信息
    return variable_data

@app.get("/api/expand/{op_id}")
def expand_operation(op_id: str):
    # 返回展开后的子图
    return subgraph_data

# 静态文件服务（前端）
app.mount("/", StaticFiles(directory="frontend/dist", html=True))
```

**前端：React + 图形库**

**图形库选择对比**：

| 库 | 优点 | 缺点 | 适用性 |
|---|---|---|---|
| **D3.js** | 灵活强大，完全可控 | 学习曲线陡峭，需要手写布局 | ⭐⭐⭐ |
| **Cytoscape.js** | 专为图可视化设计，性能好 | API 较复杂 | ⭐⭐⭐⭐⭐ |
| **React Flow** | React 友好，易用 | 功能相对简单 | ⭐⭐⭐⭐ |
| **vis.js** | 功能丰富，开箱即用 | 定制性较差 | ⭐⭐⭐ |
| **G6 (AntV)** | 阿里开源，中文文档好 | 生态相对小 | ⭐⭐⭐⭐ |

**推荐：Cytoscape.js**

- 专为复杂图设计，支持层次布局
- 性能优秀（可处理数千节点）
- 支持动态展开/折叠
- 丰富的交互 API

**前端架构**：

```
frontend/
├── src/
│   ├── components/
│   │   ├── GraphView.tsx        # 主图视图（Cytoscape）
│   │   ├── DetailPanel.tsx      # 变量详情面板
│   │   ├── ControlPanel.tsx     # 控制面板（搜索、过滤）
│   │   └── Toolbar.tsx          # 工具栏
│   ├── hooks/
│   │   ├── useGraph.ts          # 图数据管理
│   │   └── useExpand.ts         # 展开/折叠逻辑
│   ├── services/
│   │   └── api.ts               # API 调用
│   └── App.tsx
├── package.json
└── vite.config.ts
```

**核心实现示例**：

```typescript
// GraphView.tsx
import cytoscape from 'cytoscape';
import { useEffect, useRef } from 'react';

export function GraphView({ graphData }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  
  useEffect(() => {
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: graphData.elements,
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'background-color': 'data(color)',
            'shape': 'roundrectangle'
          }
        },
        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle'
          }
        }
      ],
      layout: {
        name: 'dagre',  // 层次布局
        rankDir: 'TB'
      }
    });
    
    // 点击节点展开
    cyRef.current.on('tap', 'node', async (evt) => {
      const node = evt.target;
      if (node.data('expandable')) {
        const subgraph = await fetchSubgraph(node.id());
        // 动态添加子节点
        cyRef.current.add(subgraph);
        cyRef.current.layout({ name: 'dagre' }).run();
      }
    });
    
    // 点击变量显示详情
    cyRef.current.on('tap', 'node[type="variable"]', (evt) => {
      showDetailPanel(evt.target.data());
    });
  }, [graphData]);
  
  return <div ref={containerRef} style={{ width: '100%', height: '100vh' }} />;
}
```

**参考项目**：

- **Netron**：神经网络可视化工具（<https://github.com/lutzroeder/netron）>
  - 使用 SVG + JavaScript 实现交互
  - 支持多种模型格式
- **TensorBoard Graph**：TensorFlow 的计算图可视化
  - 使用 Polymer + D3.js
  - 支持层次展开/折叠
- **NN-SVG**：神经网络架构绘制工具
  - 纯前端实现
  - 美观的样式

---

## 三、关键技术难点与解决方案

### 3.1 性能优化

**问题**：大模型的计算图可能有数千个节点，全量追踪和渲染会很慢。

**解决方案**：

1. **采样追踪**：
   - 只追踪前 N 次迭代
   - 跳过重复的子图（如循环）

2. **增量加载**：
   - 初始只加载顶层图
   - 展开时才请求子图数据

3. **虚拟化渲染**：
   - 只渲染可见区域的节点
   - 使用 LOD (Level of Detail) 技术

4. **Web Worker**：
   - 在后台线程处理图布局
   - 避免阻塞 UI

### 3.2 变量名歧义

**问题**：同一个 tensor 在不同作用域有不同名称。

**解决方案**：

1. **维护变量名映射表**：

   ```python
   variable_names = {
       id(tensor): [
           ('global', 'input_data'),
           ('SimpleNet.forward', 'x'),
           ('Linear.forward', 'input')
       ]
   }
   ```

2. **在 UI 中显示所有别名**：
   - 主名称：当前作用域的名称
   - 别名列表：其他作用域的名称

3. **提供名称追溯功能**：
   - 点击变量，显示其在整个调用链中的名称变化

### 3.3 内存占用

**问题**：记录所有变量的完整值会占用大量内存。

**解决方案**：

1. **分级存储**：
   - 小张量（< 1KB）：完整存储
   - 中等张量（1KB - 1MB）：存储统计信息 + 采样值
   - 大张量（> 1MB）：仅存储统计信息

2. **延迟加载**：
   - 运行时只记录元数据
   - 用户点击时才从 checkpoint 加载值

3. **压缩存储**：
   - 使用 NPZ 压缩格式
   - 对于重复值使用引用

---

## 四、开发路线图

### Phase 1：MVP（最小可行产品）

**目标**：实现基本的 `nn.Module` 层级可视化

**功能**：

- CLI 命令：`vode python script.py`
- 追踪 `nn.Module` 的 forward 调用
- 生成静态 SVG 图
- 显示模块名称和张量形状

**技术**：

- PyTorch hooks
- Graphviz
- JSON 存储

**参考**：torchview 的核心功能

### Phase 2：多层级追踪

**目标**：支持函数级和算子级追踪

**功能**：

- 追踪 Python 函数调用
- 追踪 `torch` 算子
- 支持选择性展开（静态图）

**技术**：

- `sys.settrace()` 或 `sys.monitoring`
- `__torch_function__` 重载
- Graphviz 的子图功能

### Phase 3：变量名和值捕获

**目标**：记录变量名和运行时值

**功能**：

- 捕获变量名（inspect）
- 记录运行时值（统计信息）
- 在静态图中显示

**技术**：

- `inspect` 模块
- NumPy 统计函数
- NPZ 存储

### Phase 4：交互式 Web 界面

**目标**：实现动态展开和详情查看

**功能**：

- Web 界面展示计算图
- 点击节点动态展开
- 点击变量查看详情
- 搜索和过滤

**技术**：

- FastAPI 后端
- React + Cytoscape.js 前端
- WebSocket 实时通信（可选）

### Phase 5：高级功能

**目标**：性能优化和高级特性

**功能**：

- 大模型性能优化
- 梯度可视化
- 执行时间分析
- 对比视图

---

## 五、技术选型总结

| 层级 | 技术选择 | 理由 |
|---|---|---|
| **CLI** | Click/Typer | 现代化、易用 |
| **追踪引擎** | PyTorch Hooks + sys.settrace | 参考 torchview + debugger |
| **变量名捕获** | inspect + AST | Python 标准库，可靠 |
| **数据存储** | JSON + NPZ | Web 友好 + 高效存储 |
| **静态渲染** | Graphviz | 成熟稳定，布局优秀 |
| **Web 后端** | FastAPI | 现代、快速、异步 |
| **Web 前端** | React + Cytoscape.js | 生态丰富 + 图可视化专业 |

---

## 六、参考资源

### 开源项目

1. **torchview** (<https://github.com/mert-kurttutan/torchview>)
   - 学习：PyTorch hooks 使用、RecorderTensor 实现
   - 复用：基础追踪框架

2. **Netron** (<https://github.com/lutzroeder/netron>)
   - 学习：模型可视化 UI 设计
   - 参考：交互方式

3. **TensorBoard** (<https://github.com/tensorflow/tensorboard>)
   - 学习：计算图可视化、数据存储格式
   - 参考：性能优化技巧

4. **Python Debugger (pdb)**
   - 学习：sys.settrace 使用、变量查看
   - 参考：frame 对象操作

### 图可视化库

1. **Cytoscape.js** (<https://js.cytoscape.org/>)
   - 文档完善，示例丰富
   - 支持层次布局、动态图

2. **React Flow** (<https://reactflow.dev/>)
   - React 集成简单
   - 适合流程图类应用

3. **G6** (<https://g6.antv.antgroup.com/>)
   - 中文文档友好
   - 阿里开源，维护活跃

### 技术文档

1. PyTorch Hooks: <https://pytorch.org/docs/stable/notes/modules.html#module-hooks>
2. Python sys.settrace: <https://docs.python.org/3/library/sys.html#sys.settrace>
3. Python inspect: <https://docs.python.org/3/library/inspect.html>
4. Graphviz DOT: <https://graphviz.org/doc/info/lang.html>

---

## 七、风险与挑战

### 7.1 技术风险

1. **性能开销**：全量追踪可能导致程序运行速度降低 10-100 倍
   - 缓解：提供采样模式、可配置追踪深度

2. **内存占用**：大模型的完整追踪数据可能达到 GB 级别
   - 缓解：分级存储、压缩、延迟加载

3. **兼容性**：不同 PyTorch 版本的 API 可能不同
   - 缓解：版本检测、适配层

### 7.2 工程挑战

1. **复杂度**：多层级追踪的实现非常复杂
   - 缓解：分阶段开发，先实现 MVP

2. **调试困难**：追踪代码本身的 bug 很难定位
   - 缓解：完善的单元测试、日志系统

3. **用户体验**：大图的交互性能和美观度难以平衡
   - 缓解：参考成熟产品（Netron、TensorBoard）

---

## 八、下一步行动

1. **搭建开发环境**
   - 创建项目结构
   - 配置开发工具（pytest、black、mypy）

2. **实现 MVP**
   - 参考 torchview 实现基础追踪
   - 生成简单的静态图

3. **验证可行性**
   - 在小模型上测试
   - 评估性能开销

4. **迭代开发**
   - 按照路线图逐步添加功能
   - 持续优化性能和用户体验
