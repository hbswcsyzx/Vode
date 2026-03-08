# Stage 2 能力边界说明

## 1. 文档目标

本文档定义 [`vode view`](vode/src/vode/cli.py:1) 的 Stage 2 能力范围，聚焦桌面浏览器中的交互式可视化查看体验。

Stage 2 的核心目标不是重新追踪程序，而是基于 Stage 1 已生成的 [`trace.json`](vode/src/vode/trace/serializer.py:54) 构建稳定、可搜索、可交互、可导出的 Web Viewer。

支持平台限定为桌面浏览器：

- Chrome
- Firefox
- Safari
- Edge

不包含移动端和平板端适配。

---

## 2. 输入与兼容范围

### 2.1 输入来源

Stage 2 只处理 Stage 1 产出的已保存追踪文件，例如：

```bash
vode view trace.json
```

输入文件应兼容 [`GraphSerializer.serialize()`](vode/src/vode/trace/serializer.py:30) 输出格式，重点依赖以下顶层结构：

```json
{
  "version": "1.0",
  "timestamp": "2026-03-07T13:20:05.123456+00:00",
  "graph": {
    "root_call_ids": ["call_0"],
    "function_calls": [],
    "variables": [],
    "edges": []
  }
}
```

### 2.2 Stage 1 JSON 兼容约束

Viewer 必须兼容以下 Stage 1 字段语义：

- `graph.root_call_ids`：根调用入口
- `graph.function_calls`：函数调用节点列表
- `graph.variables`：参数、返回值和中间数据记录
- `graph.edges`：调用树边和数据流边
- `edge.kind`：至少支持 `call_tree` 和 `dataflow`
- `FunctionCallNode.metadata`：作为扩展信息保留透传

对于未知字段，前端与后端应采用“忽略但保留兼容”的策略，不因新增字段而报错。

---

## 3. 能实现的功能

## 3.1 交互式调用树可视化

Viewer 提供调用树视图，基于 [`function_calls`](vode/src/vode/trace/serializer.py:90) 和 `call_tree` 边构建层级树。

能力包括：

- 可折叠 / 展开节点
- 显示父子调用关系
- 显示函数名、文件名、行号
- 显示参数预览、返回值预览
- 高亮当前选中节点
- 定位根调用和当前路径

适用场景：

- 查看 `main -> compute -> add` 的调用层级
- 分析递归或深层嵌套调用结构
- 定位某个函数是由谁触发的

示例：

```text
main
└── compute x=3 y=4
    ├── add a=3 b=4 -> 7
    └── pack value=7 -> tuple
```

---

## 3.2 数据流图可视化

Viewer 提供数据流视图，基于 [`variables`](vode/src/vode/trace/serializer.py:93) 与 `dataflow` 边展示对象在函数之间的传递关系。

能力包括：

- 有向图显示数据从生产者到消费者的流向
- 区分函数节点与数据连接关系
- 高亮某个节点的上游与下游
- 支持缩放、平移、框选和重置视图
- 支持按函数、模块、类型过滤图元素

适用场景：

- 查看一个返回值被哪些后续函数消费
- 识别某个 tensor 或对象的生产者函数
- 理解多层函数间的数据传播链路

示例：

```text
load_data -> preprocess -> encode -> predict
                  │
                  └────────> cache_features
```

---

## 3.3 值检查器

Viewer 提供值检查器面板，用于查看当前选中函数调用及其参数、返回值详情。

能力包括：

- 查看参数列表
- 查看返回值列表
- 显示类型名
- 显示预览文本
- 显示 tensor 元信息
- 显示 tensor 统计信息
- 显示源代码位置
- 在有数据时显示扩展元信息

适用场景：

- 检查 `x=3`、`y=4` 这类基础参数
- 查看 tuple、dict、tensor 的预览值
- 查看 PyTorch tensor 的 shape、dtype、device、numel

示例：

```text
Selected: compute
Args:
- x: int = 3
- y: int = 4
Return:
- tuple = (7, 12)
Location:
- test_cli_example.py:7
```

---

## 3.4 搜索和过滤功能

Viewer 支持面向函数调用和变量信息的搜索、过滤和快速定位。

能力包括：

- 按函数名搜索
- 按限定名搜索
- 按文件路径搜索
- 按参数预览值或返回值预览值搜索
- 按调用深度过滤
- 按模块过滤
- 按值类型过滤
- 仅显示相关子树或相关数据流

适用场景：

- 快速查找所有 `forward` 调用
- 过滤只看 `torch.nn` 模块调用
- 只查看深度小于 5 的调用树
- 搜索返回值中包含 `cuda` 或 `float32` 的节点

---

## 3.5 导出为图片或 SVG

Viewer 支持将当前视图导出为静态产物，便于分享和文档归档。

能力包括：

- 导出当前调用树视图
- 导出当前数据流视图
- 导出格式为 PNG 或 SVG
- 保留当前缩放、平移、过滤状态
- 导出前支持简单标题或文件名标注

适用场景：

- 将某段调用树粘贴到 issue 或设计文档中
- 将关键数据流图导出给团队讨论
- 为报告保留静态截图

---

## 3.6 桌面浏览器 Web 界面

Stage 2 提供本地启动的桌面 Web 界面，而不是终端内交互界面。

能力包括：

- 本地启动 HTTP 服务
- 自动在默认浏览器中打开页面
- 单页应用形式提供交互式查看
- 支持桌面端鼠标、键盘和滚轮操作
- 支持常见桌面浏览器的现代特性

适用场景：

- 本地调试程序后直接打开追踪结果
- 在大屏幕上同时查看树图、图谱和详情面板
- 配合键盘快捷键进行高频浏览

---

## 4. 不能实现的功能

Stage 2 明确不承诺以下能力。

### 4.1 不提供实时追踪

Stage 2 只能查看已经保存的 [`trace.json`](vode/src/vode/trace/serializer.py:54)，不能一边运行程序一边实时刷新图。

非支持示例：

- 程序运行时持续流式推送新节点
- 浏览器自动增量更新执行中的调用树
- 远程调试会话实时联动

### 4.2 不支持编辑或重放执行

Viewer 是只读查看器，不是执行控制器。

非支持示例：

- 在页面中修改参数后重新执行某个函数
- 点击节点回放一次历史执行
- 在图上直接编辑调用关系或数据值

### 4.3 不提供性能分析

Stage 2 不做 Stage 3 级别的性能剖析。

非支持示例：

- 火焰图
- CPU 时间分布
- GPU 时间线
- 内存热点分析
- 每个函数的性能排序面板

如果未来需要性能维度，应在后续阶段扩展 [`FunctionCallNode.metadata`](vode/src/vode/trace/serializer.py:149) 或新增专用结构。

### 4.4 不支持多文件对比

Stage 2 只聚焦单个追踪文件查看。

非支持示例：

- 两个 [`trace.json`](vode/src/vode/trace/serializer.py:54) 的差异比对
- A/B 实验调用图对比
- 同一函数在不同运行结果中的并排比较

### 4.5 不支持移动端显示

Stage 2 采用桌面优先设计，不对窄屏设备做布局保障。

非支持示例：

- 手机 Safari 上的完整面板布局
- 触屏手势优化
- 竖屏模式下的调用树操作体验保证

---

## 5. 典型使用场景

### 5.1 调试普通 Python 业务流程

用户执行：

```bash
vode trace python app.py
vode view trace.json
```

用户可在调用树中：

- 找到入口函数
- 展开子调用
- 点击某个节点查看参数与返回值
- 搜索关键函数名

### 5.2 调试 PyTorch 模型执行路径

用户希望理解某个 tensor 是如何在多个模块间流动的。

用户可在数据流图中：

- 选择某个模块调用
- 查看其输入 tensor 和输出 tensor
- 高亮下游消费节点
- 在值检查器中查看 tensor 的 shape 和 dtype

### 5.3 生成团队沟通材料

用户将当前过滤后的调用树或数据流图导出为 PNG / SVG，用于：

- issue 汇报
- 设计评审
- 调试复盘
- 文档沉淀

---

## 6. Stage 2 范围结论

Stage 2 的定位是：

> 一个面向桌面浏览器的、本地运行的、只读的 [`trace.json`](vode/src/vode/trace/serializer.py:54) 交互式查看器。

它解决的问题是：

- 让 Stage 1 已采集的数据更容易理解
- 让用户能从调用树和数据流两个视角查看程序执行
- 让值信息、搜索过滤和导出成为标准能力

它暂时不解决的问题是：

- 实时追踪
- 执行控制与重放
- 性能分析
- 多 trace 对比
- 移动端适配

该范围定义将作为 [`todo.md`](vode/docs/stage2/todo.md) 与 [`design.md`](vode/docs/stage2/design.md) 的约束基础。
