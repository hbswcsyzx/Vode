# Stage 2 技术实现计划

## 1. 文档目标

本文档给出 [`vode view`](vode/src/vode/cli.py:1) 的 Stage 2 技术实施方案，目标是在不改变 Stage 1 追踪模型的前提下，为已保存的 [`trace.json`](vode/src/vode/trace/serializer.py:54) 提供桌面端交互式可视化能力。

本文档聚焦：

- 技术栈选择
- 前后端架构
- 模块拆分
- 数据流与 API 设计
- 实施顺序
- 性能优化与风险控制

---

## 2. 目标与约束

### 2.1 Stage 2 目标

Stage 2 需要交付一个本地运行的桌面 Web Viewer，支持：

- 调用树查看
- 数据流图查看
- 节点详情检查
- 搜索和过滤
- PNG / SVG 导出

### 2.2 关键约束

- 输入数据必须兼容 Stage 1 [`GraphSerializer`](vode/src/vode/trace/serializer.py:25) 输出格式
- Viewer 为只读模式，不承担追踪职责
- 仅面向桌面浏览器
- 使用现代 Web 技术，要求 ES6+、[`React`](vode/docs/stage1/report.md:23) 18+、TypeScript
- 文档方案要可逐步实施，不依赖一次性大重构

---

## 3. 技术栈选择

## 3.1 默认推荐选型

结合 Stage 2 的交付目标、桌面工具属性和性能要求，推荐采用以下默认技术栈：

- 前端框架：React + TypeScript
- 构建工具：Vite
- UI 组件：Ant Design
- 调用树可视化：D3.js
- 数据流图可视化：Cytoscape.js
- 后端：Python FastAPI

### 3.1.1 选择理由

#### React + TypeScript

原因：

- 适合构建单页应用
- 组件化适合拆分调用树、图视图、详情面板、搜索栏
- TypeScript 便于严格描述 Stage 1 JSON 数据结构
- 生态成熟，便于后续测试与状态管理扩展

#### Vite

原因：

- 启动快、构建快
- 适合中小型前端应用快速迭代
- 对 React + TypeScript 支持成熟
- 便于最终打包静态资源给 FastAPI 托管

#### Ant Design

原因：

- 适合桌面工具式布局
- 组件完整，包含 Layout、Tree、Table、Drawer、Tooltip、Tabs、Input、Menu 等
- 能加快左侧面板、顶部栏、底部值检查器等界面的搭建速度
- 默认交互模型成熟，利于 Stage 2 快速稳定落地

#### D3.js

原因：

- 调用树是树结构，不是纯图结构
- D3.js 更适合对树节点折叠、展开、缩进、连线、路径高亮做细粒度控制
- 便于按需实现懒加载与局部更新
- 对调用树展示的视觉细节定制空间更大

#### Cytoscape.js

原因：

- 数据流图天然属于节点-边图模型
- Cytoscape.js 对缩放、平移、框选、布局和图高亮支持成熟
- 对中大型图的交互稳定性更高
- 能减少自定义图框架的工程成本

#### FastAPI

原因：

- 与 Python 主体项目一致
- 易于在 [`vode view`](vode/src/vode/cli.py:1) 命令中直接启动本地服务
- 易于提供轻量 REST API
- 可同时承担静态文件服务和 JSON 读取层

---

## 3.2 备选方案说明

虽然需求里给出 “D3.js 或 Cytoscape.js”、“Ant Design 或 shadcn/ui”，但建议不要在 Stage 2 同时保留未决方案。

推荐策略是：

- 默认 UI 采用 Ant Design
- 调用树默认采用 D3.js
- 数据流图默认采用 Cytoscape.js
- 保留未来在局部组件上替换实现的空间

这种拆分优于“一个图形库包打天下”，因为：

- 调用树和数据流图的交互本质不同
- 单库方案往往会牺牲其中一类视图体验
- 当前目标是可用性优先，而不是图形技术统一

---

## 4. 架构设计

## 4.1 总体架构

Stage 2 采用前后端分离架构，但部署方式仍然是单机本地应用。

```text
CLI: vode view trace.json
    -> Python View Entry
    -> FastAPI Server
    -> REST API + Static Assets
    -> React SPA
    -> D3 Call Tree / Cytoscape Dataflow
```

架构角色：

- Python 后端：负责加载 trace 文件、做轻量转换、提供 API、托管前端静态资源、打开浏览器
- React 前端：负责桌面交互式界面、状态管理、图形渲染、搜索和导出

### 4.1.1 为什么采用前后端分离

原因：

- 保持 [`trace.json`](vode/src/vode/trace/serializer.py:54) 解析逻辑在 Python 侧，便于复用现有模型
- 前端专注交互式体验，不承担文件系统访问职责
- 后续如果需要远程访问或扩展 API，架构可平滑升级
- 便于后续加入缓存、分页、增量查询等能力

---

## 4.2 单页应用设计

前端采用 SPA。

原因：

- 视图切换主要发生在同一份 trace 数据上
- 调用树、数据流图、值检查器共享一套状态
- 不需要多页面路由跳转
- 有利于浏览器内保持选中状态、过滤状态和导出上下文

---

## 5. 目录结构

建议目录结构如下：

```text
src/vode/view/
├── __init__.py
├── server.py          # FastAPI 服务器
├── api.py             # REST API 端点
├── loader.py          # trace.json 加载与适配
├── adapters.py        # Stage 1 JSON -> 前端视图模型转换
├── schemas.py         # API 响应模型
└── frontend/          # React 应用
    ├── src/
    │   ├── App.tsx
    │   ├── main.tsx
    │   ├── components/
    │   │   ├── CallTree.tsx
    │   │   ├── DataflowGraph.tsx
    │   │   ├── ValueInspector.tsx
    │   │   ├── SearchBar.tsx
    │   │   ├── Toolbar.tsx
    │   │   ├── FunctionList.tsx
    │   │   └── LayoutShell.tsx
    │   ├── hooks/
    │   │   ├── useGraphData.ts
    │   │   ├── useSelection.ts
    │   │   └── useFilters.ts
    │   ├── utils/
    │   │   ├── export.ts
    │   │   ├── graph.ts
    │   │   └── format.ts
    │   ├── types/
    │   │   ├── api.ts
    │   │   ├── graph.ts
    │   │   └── ui.ts
    │   └── styles/
    ├── package.json
    └── vite.config.ts
```

### 5.1 模块拆分原则

- Python 侧负责“文件和 API”
- 前端侧负责“展示和交互”
- 数据适配层与 HTTP 层分开，避免后续 API 膨胀时难维护
- 调用树组件与数据流图组件分离，避免一种渲染模型污染另一种模型

---

## 6. 核心组件设计

## 6.1 Python 后端

### 6.1.1 [`server.py`](vode/docs/stage2/todo.md)

职责：

- 创建 FastAPI 应用
- 注册 API 路由
- 托管前端静态文件
- 接收 [`vode view trace.json`](vode/src/vode/cli.py:1) 传入的 trace 路径
- 启动本地服务，默认 `localhost:8000`
- 自动打开浏览器

建议能力：

- 端口冲突时自动尝试备用端口
- 提供只绑定 `127.0.0.1` 的安全默认值
- 提供关闭浏览器自动打开的 CLI 开关

### 6.1.2 [`api.py`](vode/docs/stage2/todo.md)

职责：

- 提供 REST API
- 返回图数据、节点详情、搜索结果、统计信息
- 为前端提供最小但足够的读取接口

建议 API：

- `GET /api/graph`：获取完整图数据
- `GET /api/node/{id}`：获取节点详情
- `GET /api/search?q=...`：搜索节点
- `GET /api/stats`：获取总函数数、总变量数、最大深度等统计信息

### 6.1.3 `loader.py`

职责：

- 读取 [`trace.json`](vode/src/vode/trace/serializer.py:54)
- 调用 [`GraphSerializer.deserialize()`](vode/src/vode/trace/serializer.py:54) 或兼容逻辑
- 验证顶层结构
- 检查 `version`
- 对缺失可选字段提供默认值

### 6.1.4 `adapters.py`

职责：

- 将 Stage 1 图模型转换为前端更适合消费的结构
- 构建索引，例如 `call_by_id`、`variable_by_id`、`children_by_parent`
- 计算统计信息
- 生成树视图和图视图所需的派生字段

---

## 6.2 前端组件

### 6.2.1 [`App.tsx`](vode/docs/stage2/todo.md)

职责：

- 作为整体应用入口
- 组织页面布局
- 管理全局状态与视图切换
- 协调调用树、数据流图、值检查器和搜索栏

### 6.2.2 [`CallTree.tsx`](vode/docs/stage2/todo.md)

职责：

- 使用 D3.js 渲染树结构
- 支持折叠 / 展开
- 支持节点选中与路径高亮
- 支持局部刷新和大树优化

输入应基于：

- 根调用 ID
- 函数节点索引
- `call_tree` 边关系

### 6.2.3 [`DataflowGraph.tsx`](vode/docs/stage2/todo.md)

职责：

- 使用 Cytoscape.js 渲染数据流图
- 支持缩放、平移、节点高亮、过滤和布局切换
- 支持按当前选中节点高亮上下游

输入应基于：

- 函数节点索引
- 数据流边
- 可选变量映射信息

### 6.2.4 [`ValueInspector.tsx`](vode/docs/stage2/todo.md)

职责：

- 展示当前选中节点详情
- 显示参数、返回值、类型、预览、tensor 元信息、源代码位置
- 在未来保留扩展执行时间、异常信息等字段位置

### 6.2.5 [`SearchBar.tsx`](vode/docs/stage2/todo.md)

职责：

- 提供关键字搜索
- 提供过滤条件输入
- 支持实时搜索和结果跳转
- 支持函数名、限定名、文件路径、值预览搜索

---

## 7. API 设计

## 7.1 `GET /api/graph`

用途：

- 获取前端初始化所需的完整图数据

建议返回内容：

- `version`
- `timestamp`
- `graph`
- `indexes`
- `stats`

返回示例：

```json
{
  "version": "1.0",
  "timestamp": "2026-03-07T13:20:05.123456+00:00",
  "stats": {
    "function_count": 12,
    "variable_count": 30,
    "edge_count": 18,
    "max_depth": 4
  },
  "graph": {
    "root_call_ids": ["call_0"],
    "function_calls": [],
    "variables": [],
    "edges": []
  }
}
```

说明：

- Stage 2 初版可以一次性返回完整图
- 若后续大文件场景显著，可再扩展分页和懒加载 API

## 7.2 `GET /api/node/{id}`

用途：

- 获取指定函数节点详情
- 为值检查器提供按需查询能力

建议返回内容：

- 节点基本信息
- 参数变量详情
- 返回值变量详情
- 父子节点摘要
- 相关数据流统计

## 7.3 `GET /api/search?q=...`

用途：

- 搜索函数或变量相关内容

建议支持：

- `q`：关键字
- `kind`：可选，`call` 或 `variable`
- `depth_lte`：可选，深度过滤
- `module_prefix`：可选，模块前缀过滤

---

## 8. 数据流

Stage 2 的主数据流如下：

1. 用户运行 `vode view trace.json`
2. Python 加载并解析 JSON 文件
3. 启动 FastAPI 服务器，默认 `localhost:8000`
4. 自动打开浏览器访问 `http://localhost:8000`
5. 前端通过 API 获取图数据
6. 前端渲染交互式可视化界面
7. 用户选择节点、搜索、过滤、导出当前视图

ASCII 流程图：

```text
CLI
  -> load trace.json
  -> build indexes
  -> start FastAPI
  -> serve SPA
  -> browser requests /api/graph
  -> render call tree or dataflow
  -> inspect node details
```

---

## 9. 与 Stage 1 JSON 的兼容设计

## 9.1 直接复用字段

应直接复用以下字段，不做破坏性变更：

- `version`
- `timestamp`
- `graph.root_call_ids`
- `graph.function_calls[*].id`
- `graph.function_calls[*].parent_id`
- `graph.function_calls[*].qualified_name`
- `graph.function_calls[*].display_name`
- `graph.function_calls[*].filename`
- `graph.function_calls[*].lineno`
- `graph.function_calls[*].depth`
- `graph.function_calls[*].arg_variable_ids`
- `graph.function_calls[*].return_variable_ids`
- `graph.function_calls[*].metadata`
- `graph.variables[*]`
- `graph.edges[*].kind`

## 9.2 Viewer 侧兼容原则

- 对未知字段忽略，不报错
- 对缺失可选字段使用默认值
- 当 `version` 不匹配时给出非阻断警告
- 当 `metadata` 中存在扩展信息时，在值检查器中尽量展示

## 9.3 不建议在 Stage 2 中修改 Stage 1 输出格式

原因：

- Stage 2 应建立在 Stage 1 已稳定的序列化结构之上
- 如果 Viewer 需要派生结构，应在 `adapters.py` 中生成，而不是修改原始输出文件
- 这样可以降低 CLI、测试、文档和兼容性的连锁成本

---

## 10. 实施步骤

建议按以下顺序推进：

- [ ] 创建 FastAPI 服务器基础结构
- [ ] 实现 API 端点，加载 JSON 并提供基础图数据
- [ ] 初始化 React + TypeScript + Vite 前端项目
- [ ] 接入 Ant Design 基础桌面布局
- [ ] 实现调用树可视化，使用 D3.js
- [ ] 实现数据流图可视化，使用 Cytoscape.js
- [ ] 实现值检查器面板
- [ ] 添加搜索和过滤功能
- [ ] 实现导出功能，支持 PNG / SVG
- [ ] 完成桌面布局样式与交互细节
- [ ] 补充集成测试和用户文档

### 10.1 推荐开发顺序说明

建议先做“后端基础 + SPA 框架 + 调用树主链路”，再做“数据流图增强”。

原因：

- 调用树是最直接的基础视图
- 值检查器和搜索功能在调用树阶段就能建立主交互闭环
- 数据流图更依赖布局和高亮策略，适合放在第二阶段落地

---

## 11. 技术挑战

### 11.1 大型调用树性能优化

风险：

- 深层递归或大规模函数调用可能导致节点数过多
- 全量渲染会造成首屏卡顿

策略：

- 虚拟滚动
- 分层折叠
- 懒展开子树
- 按视口局部更新

### 11.2 复杂数据流图布局

风险：

- 边太多时容易重叠
- 力导向布局在大图场景初始化慢

策略：

- 默认采用层次布局
- 提供可切换布局策略
- 对低价值边支持过滤
- 限制首屏默认展示范围

### 11.3 前后端数据同步和缓存

风险：

- 一次性加载大 JSON 时浏览器内存压力变大
- 多组件重复派生数据会导致渲染浪费

策略：

- Python 侧提前构建索引
- 前端使用 memoization
- 统一缓存原始图数据
- 将高开销计算放在派生层或 Worker 中

### 11.4 大型 JSON 文件加载性能

风险：

- 首次打开 trace 文件耗时明显
- 搜索时可能扫描全量数据

策略：

- 服务启动时预构建索引
- 搜索优先走后端索引接口
- 对超大数据考虑懒加载和分页 API

---

## 12. 性能优化策略

Stage 2 应预留以下性能优化点：

### 12.1 虚拟滚动

- 仅渲染当前可见节点
- 用于左侧函数列表和大树面板

### 12.2 懒加载

- 默认只展开首层或关键层级
- 用户操作时再加载或计算子树显示内容

### 12.3 数据分页

- API 为未来分页查询保留扩展位
- 初版不一定启用，但接口层设计不要堵死

### 12.4 Web Worker

- 将大型图布局、搜索索引构建或导出前处理放入 Worker
- 避免阻塞主线程交互

### 12.5 增量高亮

- 仅更新选中节点及相邻节点状态
- 避免整图重新渲染

---

## 13. 测试与验收建议

## 13.1 后端测试

建议覆盖：

- 能正确读取 Stage 1 [`trace.json`](vode/src/vode/trace/serializer.py:54)
- API 返回结构稳定
- 版本不匹配时行为可控
- 缺失可选字段时不崩溃

## 13.2 前端测试

建议覆盖：

- 调用树折叠 / 展开
- 节点选择后值检查器更新
- 搜索结果定位
- 过滤后视图刷新
- 导出按钮生成有效图片

## 13.3 集成验收

验收标准应至少包括：

- 用户可通过命令启动 Viewer
- 默认浏览器可打开本地页面
- 能正确显示调用树和数据流图
- 能查看参数和返回值详情
- 能进行搜索、过滤和导出
- 在桌面浏览器下交互无明显卡死

---

## 14. 交付结论

Stage 2 的推荐技术路线是：

> FastAPI 提供本地 REST API 与静态资源服务，React + TypeScript + Vite 构建 SPA，Ant Design 提供桌面布局，D3.js 负责调用树，Cytoscape.js 负责数据流图。

这一路线的优点是：

- 与现有 Python 项目结构兼容
- 对 Stage 1 JSON 格式改动最小
- 兼顾桌面工具体验、交互流畅度与可维护性
- 能在后续阶段继续演进性能分析、扩展字段和更复杂的视图能力
