# 方案 A 详细技术方案

## 1. 方案目标

本方案采用“**函数调用图 + 参数/返回值映射**”作为 Vode 的第一阶段主路线。

目标不是完整恢复所有表达式级计算图，而是先稳定实现以下能力：

1. 零侵入启动 Python / PyTorch 程序
2. 捕获函数调用关系
3. 捕获函数边界上的输入输出对象
4. 记录对象在函数之间的来源与流向
5. 对 `torch.nn` 提供专项增强支持
6. 展示 tensor 的元信息、统计值和预览值
7. 输出静态图和交互式 Web 图

一句话总结：

> 先做一个对函数边界数据流高保真、对 `torch.nn` 场景友好的执行流可视化系统。

---

## 2. 产品定义

Vode 第一阶段的产品定义是：

> 一个面向 Python / PyTorch 的函数级执行流与数据流可视化工具。

核心观察对象是“函数调用事件”，不是表达式求值事件。

每一个函数调用都被建模为：

```text
Input -> Function Call -> Output
```

其中：

- `Input` 是调用参数
- `Function Call` 是一次具体调用事件
- `Output` 是返回值

在此基础上，再通过对象身份映射建立跨函数数据流边。

---

## 3. 非目标

当前阶段明确不做以下事情：

1. 不做通用表达式级追踪
2. 不做任意基础运算级节点化
3. 不做所有局部变量来源的稳定恢复
4. 不做底层 `torch` 算子内部实现展开
5. 不默认保存完整 tensor
6. 不承诺多进程、分布式训练的完整支持

这些能力可以在后续阶段逐步增强，但不属于本阶段主交付。

---

## 4. 核心设计原则

### 4.1 以函数调用边界为主真相来源

系统内部最可信的事件是：

1. 函数被调用了
2. 调用时有哪些参数
3. 返回时产出了什么对象
4. 当前调用的父调用是谁

这些信息来自 Python runtime，可作为图构建的主事实来源。

### 4.2 对象身份是数据流主键，变量名只是辅助展示

数据流的核心不能依赖变量名，因为变量名会变化、会重名、会失真。

系统应以“运行时对象身份 + 事件序号”作为内部索引。

变量名只用于：

1. 展示函数参数名
2. 展示返回值槽位名
3. 展示部分 frame 中的局部变量标签
4. 展示 parameter / buffer 名称

### 4.3 `torch.nn` 提供专项增强

对于 `torch.nn` 对象，系统增加专项识别：

1. `nn.Module`
2. parameter
3. buffer
4. tensor 元信息
5. module path

这样可以让神经网络场景下的函数调用图更符合用户直觉。

### 4.4 默认只记录受控值信息

值采集策略为：

1. 小 tensor：记录 preview + stats
2. 中大 tensor：记录 stats + 截断 preview
3. 超大 tensor：仅记录 stats
4. 非 tensor：记录类型名 + 安全摘要

---

## 5. 总体架构

```text
CLI Entry
  -> Runner
  -> Trace Runtime
  -> Event Collector
  -> Dataflow Resolver
  -> Graph Builder
  -> Artifact Writer
  -> Static Renderer / Web Viewer
```

### 5.1 CLI Entry

职责：

1. 解析 `vode` 自身参数
2. 识别后续 Python 命令及脚本参数
3. 注入运行时配置
4. 创建输出目录

### 5.2 Runner

职责：

1. 启动目标 Python 进程
2. 自动加载 Vode runtime
3. 维护本次 trace session 配置

### 5.3 Trace Runtime

职责：

1. 注册运行时 trace 机制
2. 捕获函数 `call` / `return`
3. 读取当前 frame 信息
4. 生成原始事件流

### 5.4 Event Collector

职责：

1. 将原始 trace 事件规范化
2. 提取参数、返回值、文件名、函数名、行号
3. 记录调用栈关系
4. 生成稳定 event id

### 5.5 Dataflow Resolver

职责：

1. 维护对象生产者映射
2. 维护对象消费者列表
3. 在事件间建立数据流边
4. 对 `torch.nn` 对象做增强标注

### 5.6 Graph Builder

职责：

1. 生成函数调用节点
2. 生成变量节点
3. 生成调用关系边
4. 生成数据流边
5. 生成前端可展开子图结构

### 5.7 Artifact Writer

职责：

1. 输出 session 元信息
2. 输出图 JSON
3. 输出 preview / stats 数据
4. 输出静态图产物

### 5.8 Viewer

职责：

1. 提供静态 SVG 渲染
2. 提供 Web 图查看器
3. 支持节点展开、搜索、详情查看

---

## 6. 运行机制

## 6.1 启动方式

```bash
vode python your_script.py [script args]
```

运行流程：

1. CLI 解析 `vode` 参数
2. 启动 Python 子进程
3. 在子进程中初始化 trace runtime
4. 执行目标脚本
5. 采集事件并写出图数据
6. 根据配置生成静态图或启动本地 Web 服务

### 6.2 运行模式

第一阶段支持两种模式：

1. **wrap mode**
   - 用户只改启动命令
   - 默认模式

2. **api mode**
   - 用户显式包裹一段代码或调用函数
   - 适用于复杂场景、调试和精确边界控制

虽然产品追求零侵入，但保留 `api mode` 可以显著提升复杂场景下的可控性。

---

## 7. Trace 事件设计

### 7.1 事件类型

第一阶段最核心的事件只有两类：

1. `call`
2. `return`

可选辅助事件：

1. `exception`
2. `line` 仅用于调试，不进入默认主流程

### 7.2 `call` 事件记录内容

每次函数调用至少记录：

1. `event_id`
2. `event_type = call`
3. `parent_event_id`
4. `call_depth`
5. `filename`
6. `qualified_function_name`
7. `lineno`
8. `arg_bindings`
9. `frame_locals_snapshot_summary` 可选
10. `timestamp`

### 7.3 `return` 事件记录内容

每次函数返回至少记录：

1. `event_id`
2. `event_type = return`
3. `call_event_id`
4. `return_value_ref`
5. `timestamp`
6. `exception_info` 可选

### 7.4 调用栈维护

trace runtime 必须维护调用栈：

1. `call` 时入栈
2. `return` 时出栈
3. 当前栈顶作为后续调用的 `parent_event_id`

这样可以形成稳定的调用树。

---

## 8. frame 读取策略

### 8.1 读取内容

每次函数 `call` 时，读取：

1. `frame.f_code.co_filename`
2. `frame.f_code.co_name`
3. `frame.f_lineno`
4. `frame.f_locals`
5. `frame.f_globals` 仅在必要时访问

### 8.2 参数提取

参数提取优先使用函数签名绑定结果，而不是直接全量依赖 `locals`。

记录形式示例：

```text
arg_bindings = {
  input: obj_ref_1,
  hidden: obj_ref_2,
}
```

这样可以保留函数边界上最重要的语义信息。

### 8.3 返回值提取

在 `return` 事件中记录：

1. 返回对象引用
2. 若返回值为 tuple/list/dict，则标准化为槽位形式

例如：

- `return.0`
- `return.1`
- `return.logits`

---

## 9. 对象引用与数据流解析

## 9.1 为什么需要对象引用层

如果只记录变量名，将无法稳定建立跨函数数据流。

因此必须引入对象引用层：

```text
ObjectRef = runtime object identity + event scoped metadata
```

### 9.2 ObjectRef 建议结构

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ObjectRef:
    runtime_id: int
    type_name: str
    is_tensor: bool
    metadata: dict[str, Any]
```

说明：

- `runtime_id` 可基于单次执行中的 `id(obj)`
- 但仅用于单次 session 内部索引
- 不能作为跨运行稳定 ID

### 9.3 生产者映射

维护：

```text
produced_by[runtime_id] = return_event_id
```

含义是：

- 某个对象由哪个函数返回产生

### 9.4 消费者映射

维护：

```text
consumed_by[runtime_id] = [call_event_id_1, call_event_id_2, ...]
```

含义是：

- 某个对象后续被哪些函数作为输入消费

### 9.5 数据流边生成规则

如果：

1. 函数 A 的返回值对象 runtime id 为 `r1`
2. 函数 B 的参数中出现了同一个 `r1`

则建立一条数据流边：

```text
A.return -> B.arg_x
```

这就是函数边界上的来源与流向。

---

## 10. 输入输出标准化

### 10.1 为什么要标准化

函数参数和返回值可能是：

1. 单一对象
2. tuple
3. list
4. dict
5. 嵌套结构

为了统一前端和存储结构，必须将其规范化为槽位路径。

### 10.2 标准化规则

示例：

- `arg.0`
- `arg.input`
- `arg.hidden.0`
- `return.0`
- `return.logits`
- `return.hidden_states.2`

### 10.3 展示策略

前端展示时：

1. 默认展示首层槽位
2. 深层结构在详情面板展开
3. 不在主图中完整铺开大型嵌套容器

---

## 11. `torch.nn` 专项支持

## 11.1 目标

在函数调用主图基础上，为 PyTorch 神经网络场景增加额外语义信息。

### 11.2 参考来源

实现思路可参考 [`torchview`](../../torchview/README.md) 以及其源码中的模块追踪与 tensor 处理方法，例如：

- [`torchview/src/torchview/torchview.py`](../../torchview/src/torchview/torchview.py)
- [`torchview/src/torchview/recorder_tensor.py`](../../torchview/src/torchview/recorder_tensor.py)
- [`torchview/src/torchview/computation_graph.py`](../../torchview/src/torchview/computation_graph.py)

### 11.3 支持内容

对于以下对象增加专项识别：

1. `nn.Module`
2. `nn.Parameter`
3. buffer
4. tensor
5. module method，如 `Linear.forward`

### 11.4 模块路径识别

如果当前调用与某个 module method 相关，则记录：

1. module class
2. module path
3. owned parameters
4. owned buffers

例如：

```text
module_class = Linear
module_path = encoder.layers.0.self_attn.q_proj
```

### 11.5 Parameter / Buffer 展示

对于参数和 buffer，记录：

1. 名称
2. shape
3. dtype
4. device
5. 所属 module path

并在函数详情面板中展示“该调用关联了哪些参数/缓冲区”。

### 11.6 为什么不以 hook 作为主路径

虽然 [`torchview`](../../torchview/README.md) 的 hook 机制非常适合模块图，但本方案的主视图是函数调用图。

因此：

- hook 是专项增强手段
- trace 事件才是主事实来源

也就是说：

1. 用函数 trace 建主骨架
2. 用 `torch.nn` 识别和 hook 能力补充语义信息

---

## 12. 值采集策略

### 12.1 Tensor 元信息

对 tensor 记录：

1. shape
2. dtype
3. device
4. requires_grad
5. numel

### 12.2 统计值

默认记录：

1. min
2. max
3. mean
4. std

### 12.3 Preview 策略

建议：

1. `numel <= 32`：记录较完整 preview
2. `32 < numel <= 4096`：记录前若干值 + stats
3. `numel > 4096`：只记录 stats

### 12.4 非 tensor 对象

对非 tensor：

1. 记录 `type(obj).__name__`
2. 记录安全摘要字符串
3. 长摘要截断
4. 避免深层递归序列化

### 12.5 安全与性能要求

任何值采集都必须满足：

1. 不导致程序显著额外副作用
2. 不默认把大 tensor 全量转到 CPU
3. 失败时可降级为只记录类型和 meta

---

## 13. 数据模型

### 13.1 核心实体

```python
from dataclasses import dataclass, field
from typing import Any, Literal

NodeKind = Literal['function_call', 'variable', 'parameter', 'buffer']
EdgeKind = Literal['call_tree', 'dataflow', 'owns']

@dataclass
class TensorMeta:
    shape: list[int] | None
    dtype: str | None
    device: str | None
    requires_grad: bool | None
    numel: int | None

@dataclass
class TensorStats:
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None

@dataclass
class ValuePreview:
    text: str | None = None
    data: Any | None = None

@dataclass
class VariableRecord:
    id: str
    slot_path: str
    display_name: str
    runtime_object_id: int | None
    type_name: str
    tensor_meta: TensorMeta | None
    tensor_stats: TensorStats | None
    preview: ValuePreview | None
    producer_call_id: str | None
    consumer_call_ids: list[str] = field(default_factory=list)

@dataclass
class FunctionCallNode:
    id: str
    parent_id: str | None
    qualified_name: str
    display_name: str
    filename: str
    lineno: int
    depth: int
    arg_variable_ids: list[str] = field(default_factory=list)
    return_variable_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphEdge:
    id: str
    src_id: str
    dst_id: str
    kind: EdgeKind

@dataclass
class TraceGraph:
    root_call_ids: list[str]
    function_calls: list[FunctionCallNode]
    variables: list[VariableRecord]
    edges: list[GraphEdge]
```

### 13.2 ID 规则

建议：

1. 调用节点：`call:{seq}`
2. 变量节点：`var:{call_id}:{slot_path}`
3. 参数节点：`param:{module_path}:{name}`
4. buffer 节点：`buffer:{module_path}:{name}`
5. 边：`edge:{src}->{dst}:{kind}`

---

## 14. 图构建方案

### 14.1 图结构

主图使用两类关系：

1. 调用树关系
2. 数据流关系

调用树关系表示：

- 哪个函数内部调用了哪个子函数

数据流关系表示：

- 哪个调用返回的对象流入了后续哪个调用的参数槽位

### 14.2 展开语义

UI 的展开/折叠针对函数调用节点进行：

1. 默认展示根调用和第一层子调用
2. 点击某个调用节点时，加载其直属子调用及相关变量
3. 不一次性摊平整个图

### 14.3 变量展示策略

主图中的变量节点仅展示：

1. 输入变量
2. 返回变量
3. 关键 parameter / buffer

详情面板再展示：

1. 完整类型信息
2. tensor 统计值
3. preview
4. 生产者 / 消费者列表

---

## 15. 存储格式

### 15.1 输出目录

```text
.vode/
  session.json
  graph/
    calls.json
    variables.json
    edges.json
  values/
    previews.json
  render/
    graph.svg
  web/
    index.html
```

### 15.2 文件内容

#### `session.json`

保存：

1. session id
2. command
3. python version
4. torch version
5. trace config
6. start / end time

#### `calls.json`

保存所有函数调用节点。

#### `variables.json`

保存变量节点和详细元信息。

#### `edges.json`

保存调用树边和数据流边。

#### `previews.json`

保存较大的 preview 数据，避免主图文件过重。

---

## 16. 静态渲染方案

### 16.1 技术选择

静态渲染使用 Graphviz。

### 16.2 静态图目标

第一阶段静态图重点展示：

1. 函数调用层级
2. 关键输入输出变量
3. 关键数据流向
4. PyTorch 调用的 module 名称和 shape

### 16.3 不追求的效果

1. 所有节点一次性都清晰可读
2. 完整替代 Web 交互视图
3. 超大图的完美自动布局

---

## 17. Web 交互式方案

### 17.1 技术栈

1. 后端：FastAPI
2. 前端：React
3. 图组件：Cytoscape.js

### 17.2 页面组成

1. 图主视图
2. 调用详情面板
3. 变量详情面板
4. 搜索和过滤面板

### 17.3 必要交互

1. 点击函数节点展开子调用
2. 点击变量节点查看统计值和 preview
3. 搜索函数名、文件路径、module path
4. 按节点类型过滤显示
5. 按文件路径过滤用户代码 / 第三方代码

### 17.4 API 设计

#### `GET /api/session`

返回 session 元信息。

#### `GET /api/graph/root`

返回根调用图。

#### `GET /api/call/{call_id}`

返回某个调用节点的直属子图。

#### `GET /api/variable/{variable_id}`

返回变量详情。

#### `GET /api/search?q=...`

搜索函数名、模块路径、文件名。

---

## 18. CLI 设计

### 18.1 命令形式

```bash
vode [vode options] python your_script.py [script args]
```

### 18.2 第一阶段参数

```text
--output-dir PATH
--format svg|web|all
--max-depth INT
--value-policy none|preview|auto
--include-third-party / --exclude-third-party
--include-torch-nn / --exclude-torch-nn
--serve
```

### 18.3 参数说明

- `--output-dir`：指定输出目录
- `--format`：控制输出类型
- `--max-depth`：限制调用树深度
- `--value-policy`：控制值采集强度
- `--include-third-party`：是否包含第三方库函数
- `--include-torch-nn`：是否启用 `torch.nn` 增强
- `--serve`：生成后启动本地服务

---

## 19. 过滤与降噪策略

### 19.1 为什么必须降噪

函数级 trace 天然会产生大量噪音。

因此第一阶段必须强制设计过滤策略。

### 19.2 默认过滤规则

默认只保留：

1. 用户工作区代码函数
2. `torch.nn` 相关关键调用
3. 少量显式白名单函数

默认排除：

1. 标准库内部函数
2. site-packages 大量无关函数
3. 调试器自身内部函数

### 19.3 可配置策略

用户可以按以下维度过滤：

1. 文件路径
2. 模块路径
3. 调用深度
4. 对象类型
5. 是否显示第三方函数

---

## 20. 风险与控制措施

### 20.1 性能风险

函数级 trace 本身会带来明显开销。

控制措施：

1. 先只做 `call` / `return`
2. 默认不开 `line` 级追踪
3. 默认过滤第三方代码
4. 限制 preview 与 stats 计算范围

### 20.2 数据量风险

复杂程序会产生大量调用事件和变量节点。

控制措施：

1. 最大深度限制
2. 最大节点数限制
3. 大值只保留 stats
4. Web 按需加载子图

### 20.3 来源误判风险

对象流向建立依赖对象身份映射，在 alias、原地修改场景下可能出现歧义。

控制措施：

1. 明确当前阶段只承诺函数级来源
2. 对原地修改对象增加 warning 标记
3. 在 UI 中显示“近似来源”而非绝对真相

### 20.4 `torch.nn` 兼容性风险

不同模型写法可能让模块语义识别不完整。

控制措施：

1. 参考 [`torchview`](../../torchview/README.md) 的模块处理经验
2. 对常见模型结构建立测试集
3. 对识别失败的场景回退为普通函数节点

---

## 21. 分阶段实施计划

## Phase 0：最小验证

### 目标

证明“函数 trace + 参数/返回值映射 + 基础数据流边”可跑通。

### 交付物

1. 最小 trace runtime
2. 最小事件流 JSON
3. 单个示例程序的函数调用图

### 验收标准

1. 能记录函数 `call` / `return`
2. 能显示参数和返回值
3. 能建立至少一条跨函数数据流边

---

## Phase 1：函数级 MVP

### 目标

交付可用的函数调用图可视化原型。

### 功能范围

1. CLI 启动
2. 函数调用树
3. 参数/返回值映射
4. 数据流边
5. tensor meta / stats / preview
6. 静态 SVG 输出
7. 基础 Web 查看器

### 验收标准

1. 能在小型 Python / PyTorch 程序上运行
2. 能展开函数调用层级
3. 能查看变量统计值和 preview
4. 能展示函数边界上的来源与流向

---

## Phase 2：`torch.nn` 增强版

### 目标

让神经网络场景具有更强解释力。

### 功能范围

1. 识别 `nn.Module` 调用
2. 展示 module path
3. 展示 parameter / buffer
4. 与函数调用图关联显示
5. 优化神经网络场景下的布局与过滤

### 验收标准

1. 常见 `nn.Module` 模型可读性显著提升
2. 参数与模块归属关系可展示
3. PyTorch 用户能通过图理解主要前向数据流

---

## Phase 3：实验性增强

### 目标

在不破坏主架构的前提下，尝试局部增强来源推断能力。

### 范围

1. 局部 frame diff
2. AST 辅助推断
3. 白名单函数更细粒度展示

### 明确说明

这一阶段是实验性增强，不作为第一阶段主承诺。

---

## 22. 测试策略

### 22.1 单元测试

覆盖：

1. 事件规范化
2. 参数绑定提取
3. 返回值标准化
4. 对象引用构建
5. 数据流边生成
6. tensor meta / stats / preview 提取

### 22.2 集成测试

测试对象包括：

1. 普通函数嵌套调用
2. 多输入多输出函数
3. 含 tuple / dict 返回值函数
4. PyTorch 小模型
5. 含 `nn.Module` 与 parameter 的模型

### 22.3 回归测试

固定 golden JSON：

1. 调用节点数
2. 数据流边数
3. 关键字段
4. SVG 关键片段

### 22.4 性能测试

至少记录：

1. trace 总耗时
2. 单次运行节点数
3. graph JSON 大小
4. stats 计算开销
5. Web 首屏加载耗时

---

## 23. 最终结论

方案 A 是当前最适合作为 Vode 第一阶段落地方案的路线。

它的优势在于：

1. 可行性高
2. 运行时信息来源清晰
3. 数据流主线明确
4. 可以与 `torch.nn` 特殊支持结合
5. 能较好覆盖用户最关心的“函数之间数据如何流动”问题

它的边界也很明确：

1. 当前只保证函数级来源
2. 不保证表达式级来源
3. 不展开到底层算子内部实现
4. 不记录完整 tensor

因此，这个方案不是终局，但它是一个**工程上可信、产品上有价值、后续可持续增强**的起点。
