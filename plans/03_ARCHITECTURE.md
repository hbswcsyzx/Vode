# Vode 项目架构

## 完整项目结构

```
vode/
├── README.md                      # 项目说明
├── LICENSE                        # 开源协议
├── pyproject.toml                 # Poetry 配置
├── setup.py                       # 安装脚本
├── .gitignore
├── .voderc.example                # 配置文件示例
│
├── docs/                          # 文档
│   ├── index.md                   # 文档首页
│   ├── quickstart.md              # 快速开始
│   ├── user_guide/                # 用户指南
│   │   ├── function_flow.md
│   │   ├── computation_flow.md
│   │   ├── static_export.md
│   │   └── interactive_view.md
│   ├── api_reference/             # API 文档
│   │   ├── capture.md
│   │   ├── visualize.md
│   │   └── cli.md
│   ├── examples/                  # 示例
│   │   ├── basic_usage.md
│   │   ├── pytorch_model.md
│   │   └── performance_analysis.md
│   └── development/               # 开发文档
│       ├── architecture.md
│       ├── contributing.md
│       └── testing.md
│
├── src/                           # 源代码
│   └── vode/
│       ├── __init__.py            # 包初始化，导出公共 API
│       ├── __main__.py            # 入口点：python -m vode
│       ├── cli.py                 # CLI 命令定义
│       ├── config.py              # 配置管理
│       │
│       ├── capture/               # 捕获模块
│       │   ├── __init__.py
│       │   ├── base.py            # 基类：BaseTracer
│       │   ├── function_tracer.py # 函数流捕获
│       │   ├── computation_tracer.py # 计算流捕获
│       │   ├── recorder_tensor.py # RecorderTensor 实现
│       │   ├── hooks.py           # PyTorch hooks 工具
│       │   └── utils.py           # 捕获工具函数
│       │
│       ├── core/                  # 核心数据结构
│       │   ├── __init__.py
│       │   ├── graph.py           # Graph, Node, Edge 类
│       │   ├── serializer.py      # JSON 序列化/反序列化
│       │   ├── validator.py       # 数据验证
│       │   └── types.py           # 类型定义
│       │
│       ├── visualize/             # 可视化模块
│       │   ├── __init__.py
│       │   ├── graphviz_renderer.py # Graphviz 静态导出
│       │   ├── server.py          # FastAPI 服务器
│       │   ├── routes.py          # API 路由
│       │   └── static/            # 前端构建产物（由 frontend/ 构建）
│       │       ├── index.html
│       │       ├── assets/
│       │       │   ├── index.js
│       │       │   └── index.css
│       │       └── favicon.ico
│       │
│       ├── editor/                # 可视化编辑器（未来）
│       │   ├── __init__.py
│       │   ├── server.py          # 编辑器服务器
│       │   ├── codegen.py         # 代码生成器
│       │   ├── templates/         # 模型模板
│       │   │   ├── resnet.json
│       │   │   ├── vgg.json
│       │   │   └── transformer.json
│       │   └── static/            # 编辑器前端
│       │
│       └── utils/                 # 通用工具
│           ├── __init__.py
│           ├── logger.py          # 日志
│           ├── file_utils.py      # 文件操作
│           └── tensor_utils.py    # 张量工具
│
├── frontend/                      # 前端项目（交互式查看器）
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   │
│   ├── public/                    # 静态资源
│   │   └── favicon.ico
│   │
│   └── src/
│       ├── main.tsx               # 入口文件
│       ├── App.tsx                # 根组件
│       ├── vite-env.d.ts
│       │
│       ├── components/            # React 组件
│       │   ├── GraphView/
│       │   │   ├── index.tsx      # React Flow 主视图
│       │   │   ├── CustomNode.tsx # 自定义节点
│       │   │   ├── CustomEdge.tsx # 自定义边
│       │   │   └── styles.css
│       │   │
│       │   ├── NodeDetailsPanel/
│       │   │   ├── index.tsx      # 详情面板
│       │   │   ├── FunctionDetails.tsx
│       │   │   ├── ComputationDetails.tsx
│       │   │   └── styles.css
│       │   │
│       │   ├── Toolbar/
│       │   │   ├── index.tsx      # 工具栏
│       │   │   ├── SearchBar.tsx
│       │   │   ├── FilterPanel.tsx
│       │   │   └── styles.css
│       │   │
│       │   ├── MiniMap/
│       │   │   ├── index.tsx
│       │   │   └── styles.css
│       │   │
│       │   └── common/            # 通用组件
│       │       ├── Button.tsx
│       │       ├── Input.tsx
│       │       └── Modal.tsx
│       │
│       ├── hooks/                 # 自定义 Hooks
│       │   ├── useGraph.ts        # 加载图数据
│       │   ├── useNodeExpansion.ts # 展开/折叠逻辑
│       │   ├── useSearch.ts       # 搜索功能
│       │   ├── useFilter.ts       # 过滤功能
│       │   └── useLayout.ts       # 布局算法
│       │
│       ├── stores/                # 状态管理（Zustand）
│       │   ├── graphStore.ts      # 图数据状态
│       │   ├── uiStore.ts         # UI 状态
│       │   └── settingsStore.ts   # 设置状态
│       │
│       ├── types/                 # TypeScript 类型
│       │   ├── graph.ts           # 图数据类型
│       │   ├── node.ts            # 节点类型
│       │   └── api.ts             # API 类型
│       │
│       ├── api/                   # API 调用
│       │   ├── client.ts          # HTTP 客户端
│       │   ├── graph.ts           # 图数据 API
│       │   └── node.ts            # 节点 API
│       │
│       ├── utils/                 # 工具函数
│       │   ├── layout.ts          # 布局算法
│       │   ├── format.ts          # 格式化工具
│       │   └── export.ts          # 导出工具
│       │
│       └── styles/                # 全局样式
│           ├── global.css
│           ├── variables.css
│           └── themes/
│               ├── light.css
│               └── dark.css
│
├── editor-frontend/               # 编辑器前端（未来）
│   ├── package.json
│   └── src/
│       ├── components/
│       │   ├── ModuleLibrary/     # 模块库面板
│       │   ├── Canvas/            # 画布
│       │   ├── PropertyPanel/     # 属性面板
│       │   └── CodePreview/       # 代码预览
│       └── ...
│
├── tests/                         # 测试
│   ├── __init__.py
│   ├── conftest.py                # pytest 配置
│   │
│   ├── unit/                      # 单元测试
│   │   ├── test_function_tracer.py
│   │   ├── test_computation_tracer.py
│   │   ├── test_graph.py
│   │   ├── test_serializer.py
│   │   └── test_renderer.py
│   │
│   ├── integration/               # 集成测试
│   │   ├── test_trace_export.py
│   │   ├── test_trace_view.py
│   │   └── test_end_to_end.py
│   │
│   ├── performance/               # 性能测试
│   │   ├── test_large_graph.py
│   │   ├── test_deep_recursion.py
│   │   └── test_memory.py
│   │
│   └── fixtures/                  # 测试数据
│       ├── simple_function.py
│       ├── simple_model.py
│       ├── complex_model.py
│       └── expected_outputs/
│           ├── simple_function.json
│           └── simple_model.json
│
├── examples/                      # 示例脚本
│   ├── README.md
│   ├── 01_simple_function.py      # 简单函数示例
│   ├── 02_pytorch_model.py        # PyTorch 模型示例
│   ├── 03_complex_workflow.py     # 复杂工作流示例
│   ├── 04_performance_analysis.py # 性能分析示例
│   └── outputs/                   # 示例输出
│       ├── simple_function.vode.json
│       ├── simple_function.png
│       └── pytorch_model.vode.json
│
├── scripts/                       # 开发脚本
│   ├── build_frontend.sh          # 构建前端
│   ├── run_tests.sh               # 运行测试
│   ├── generate_docs.sh           # 生成文档
│   └── release.sh                 # 发布脚本
│
├── benchmarks/                    # 性能基准测试
│   ├── benchmark_tracer.py
│   ├── benchmark_renderer.py
│   └── results/
│
└── .github/                       # GitHub 配置
    ├── workflows/
    │   ├── ci.yml                 # CI 流程
    │   ├── release.yml            # 发布流程
    │   └── docs.yml               # 文档部署
    ├── ISSUE_TEMPLATE/
    └── PULL_REQUEST_TEMPLATE.md
```

---

## 核心模块详解

### 1. 捕获模块 (`src/vode/capture/`)

**职责**：捕获程序执行轨迹，生成图数据。

**关键文件**：

- `base.py`: 定义 `BaseTracer` 抽象基类
- `function_tracer.py`: 使用 `sys.settrace()` 捕获函数调用
- `computation_tracer.py`: 使用 hooks 捕获计算图
- `recorder_tensor.py`: `RecorderTensor` 子类实现

**数据流**：

```
Python 程序 → Tracer → Graph 对象 → JSON 文件
```

---

### 2. 核心模块 (`src/vode/core/`)

**职责**：定义核心数据结构，处理序列化。

**关键文件**：

- `graph.py`: `Graph`, `Node`, `Edge` 数据类
- `serializer.py`: JSON 序列化/反序列化
- `validator.py`: 数据验证（确保 JSON 格式正确）
- `types.py`: 类型定义（`NodeType`, `EdgeType` 等）

**数据流**：

```
Graph 对象 ↔ JSON 字符串 ↔ 文件
```

---

### 3. 可视化模块 (`src/vode/visualize/`)

**职责**：静态导出和交互式查看。

**关键文件**：

- `graphviz_renderer.py`: 使用 Graphviz 生成图片
- `server.py`: FastAPI 服务器
- `routes.py`: API 路由定义
- `static/`: 前端构建产物

**数据流**：

```
# 静态导出
JSON 文件 → GraphvizRenderer → PNG/SVG/PDF

# 交互式查看
JSON 文件 → FastAPI → REST API → 前端 → 浏览器
```

---

### 4. 前端模块 (`frontend/`)

**职责**：交互式可视化界面。

**技术栈**：

- React 18 + TypeScript
- React Flow（图形库）
- Ant Design（UI 组件）
- Zustand（状态管理）
- Vite（构建工具）

**组件层级**：

```
App
├── Toolbar（搜索、过滤、控制）
├── GraphView（React Flow 主视图）
│   ├── CustomNode（自定义节点）
│   └── CustomEdge（自定义边）
├── NodeDetailsPanel（详情面板）
│   ├── FunctionDetails
│   └── ComputationDetails
└── MiniMap（小地图）
```

**状态管理**：

```typescript
// graphStore.ts
{
  graph: Graph,              // 完整图数据
  visibleNodes: Node[],      // 当前可见节点
  expandedNodes: Set<string>, // 已展开节点
  selectedNode: Node | null   // 选中节点
}

// uiStore.ts
{
  searchQuery: string,
  filters: FilterConfig,
  theme: 'light' | 'dark',
  sidebarOpen: boolean
}
```

---

### 5. CLI 模块 (`src/vode/cli.py`)

**职责**：命令行接口。

**命令结构**：

```
vode
├── trace       # 捕获执行轨迹
├── export      # 导出静态图片
├── view        # 交互式查看
└── editor      # 可视化编辑器（未来）
```

**实现方式**：

```python
import click

@click.group()
def cli():
    """Vode: View Your Code"""
    pass

@cli.command()
@click.argument('script')
@click.option('--mode', type=click.Choice(['function', 'computation']))
def trace(script, mode):
    # 实现捕获逻辑
    pass

# 其他命令...
```

---

## 数据流图

### 完整工作流

```
┌─────────────┐
│ Python 脚本 │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  vode trace     │ ← 捕获模块
│  (Tracer)       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  .vode.json     │ ← JSON 文件
└──────┬──────────┘
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ vode export  │  │  vode view   │  │ 直接读取     │
│ (Graphviz)   │  │  (FastAPI)   │  │ (Python API) │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ PNG/SVG/PDF  │  │   浏览器     │  │  自定义处理  │
└──────────────┘  │ (React Flow) │  └──────────────┘
                  └──────────────┘
```

---

## 模块依赖关系

```
cli.py
  ├─→ capture/
  │     ├─→ core/
  │     └─→ utils/
  │
  ├─→ visualize/
  │     ├─→ core/
  │     ├─→ graphviz_renderer.py
  │     └─→ server.py
  │           └─→ frontend/ (static files)
  │
  └─→ editor/ (未来)
        ├─→ core/
        └─→ editor-frontend/ (static files)
```

**依赖原则**：

- `core/` 不依赖任何其他模块（纯数据结构）
- `capture/` 只依赖 `core/` 和 `utils/`
- `visualize/` 依赖 `core/`
- `cli.py` 协调所有模块

---

## 配置文件 (`.voderc`)

用户可以在项目根目录创建 `.voderc` 配置文件：

```json
{
  "capture": {
    "function": {
      "max_depth": 10,
      "stop_patterns": [
        "torch\\.nn\\.modules\\..*\\.forward",
        "torch\\.nn\\.functional\\..*"
      ],
      "capture_data": true
    },
    "computation": {
      "capture_data": true,
      "capture_gradients": false
    }
  },
  "export": {
    "default_format": "svg",
    "default_depth": 3,
    "theme": "light",
    "layout": "TB"
  },
  "view": {
    "port": 8000,
    "auto_open_browser": true,
    "theme": "dark"
  }
}
```

---

## 扩展机制

### 插件系统（未来）

允许用户自定义捕获器和渲染器：

```python
# 自定义捕获器
from vode.capture import BaseTracer

class MyCustomTracer(BaseTracer):
    def trace(self, target):
        # 自定义捕获逻辑
        pass

# 注册插件
vode.register_tracer('custom', MyCustomTracer)

# 使用
$ vode trace --mode custom script.py
```

---

## 部署结构

### PyPI 包结构

```
vode-0.1.0/
├── src/vode/
│   ├── __init__.py
│   ├── cli.py
│   ├── capture/
│   ├── core/
│   ├── visualize/
│   │   └── static/  ← 包含前端构建产物
│   └── utils/
├── README.md
├── LICENSE
└── pyproject.toml
```

### 安装后的文件位置

```
site-packages/
└── vode/
    ├── __init__.py
    ├── cli.py
    ├── capture/
    ├── core/
    ├── visualize/
    │   └── static/
    └── utils/

bin/
└── vode  ← CLI 入口点
```

---

## 开发工作流

### 1. 开发环境搭建

```bash
# 克隆仓库
git clone https://github.com/your-org/vode.git
cd vode

# 安装 Python 依赖
poetry install

# 安装前端依赖
cd frontend
npm install
cd ..
```

### 2. 开发模式

```bash
# 后端开发（自动重载）
poetry run python -m vode view examples/simple.vode.json --reload

# 前端开发（热更新）
cd frontend
npm run dev
```

### 3. 构建

```bash
# 构建前端
cd frontend
npm run build  # 输出到 src/vode/visualize/static/

# 构建 Python 包
poetry build
```

### 4. 测试

```bash
# 运行所有测试
poetry run pytest

# 运行特定测试
poetry run pytest tests/unit/test_function_tracer.py

# 覆盖率报告
poetry run pytest --cov=vode --cov-report=html
```

### 5. 发布

```bash
# 更新版本号
poetry version patch  # 或 minor, major

# 构建
poetry build

# 发布到 PyPI
poetry publish
```

---

## 总结

这个架构设计遵循以下原则：

1. **模块化**：清晰的模块划分，职责单一
2. **可扩展**：插件机制，易于添加新功能
3. **可测试**：每个模块独立测试
4. **用户友好**：简单的 CLI，丰富的配置选项
5. **性能优先**：懒加载，增量渲染，缓存机制

通过这个架构，Vode 可以从简单的捕获工具逐步演进为功能完整的可视化开发平台。
