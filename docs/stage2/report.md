# Stage 2 完成报告

## 1. 实现概述

**实现日期**: 2026-03-07  
**版本**: 0.1.0  
**状态**: Stage 2 核心功能已完成

Stage 2 为 Vode 追踪系统实现了基于 Web 的交互式可视化查看器，支持在桌面浏览器中查看和分析 Stage 1 生成的追踪文件。

## 2. 技术栈

### 2.1 后端

- **FastAPI**: REST API 服务器
- **Uvicorn**: ASGI 服务器
- **Python 3.10+**: 运行环境

### 2.2 前端

- **React 18**: UI 框架
- **TypeScript**: 类型安全
- **Vite**: 构建工具
- **Ant Design**: UI 组件库
- **D3.js**: 调用树可视化
- **Cytoscape.js**: 数据流图可视化

## 3. 已实现功能

### 3.1 后端服务

- ✅ FastAPI 服务器 ([`vode/src/vode/view/server.py`](vode/src/vode/view/server.py))
- ✅ REST API 端点 ([`vode/src/vode/view/api.py`](vode/src/vode/view/api.py))
  - `GET /api/graph` - 获取完整图数据
  - `GET /api/node/:id` - 获取节点详情
  - `GET /api/search?q=...` - 搜索节点
  - `GET /api/stats` - 获取统计信息
- ✅ 静态文件服务
- ✅ CORS 中间件配置
- ✅ 自动打开浏览器

### 3.2 前端应用

- ✅ React + TypeScript 单页应用
- ✅ 调用树可视化组件 (D3.js)
- ✅ 数据流图可视化组件 (Cytoscape.js)
- ✅ 值检查器面板
- ✅ 搜索和过滤功能
- ✅ 响应式布局

### 3.3 CLI 集成

- ✅ `vode view` 命令 ([`vode/src/vode/cli.py`](vode/src/vode/cli.py))
- ✅ `--web` 标志启动 Web 查看器
- ✅ `--host` 和 `--port` 配置选项
- ✅ `--no-browser` 禁用自动打开浏览器
- ✅ 默认文本输出模式

## 4. 文件结构

```
vode/src/vode/view/
├── __init__.py
├── server.py          # FastAPI 服务器
├── api.py             # REST API 端点
└── frontend/          # React 应用
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── types/graph.ts
        ├── utils/api.ts
        ├── hooks/useGraphData.ts
        └── components/
            ├── Header.tsx
            ├── LeftPanel.tsx
            ├── MainView.tsx
            ├── CallTreeView.tsx
            ├── DataflowView.tsx
            └── ValueInspector.tsx
```

## 5. 使用方法

### 5.1 首次设置

```bash
# 1. 安装 Vode
cd vode
pip install -e .

# 2. 构建前端（首次使用）
cd src/vode/view/frontend
npm install
npm run build
cd ../../../..
```

### 5.2 基本使用

```bash
# 1. 生成追踪文件
python -m vode trace example.py -o trace.json

# 2. 使用 Web 查看器
python -m vode view trace.json --web

# 3. 或使用文本输出（默认）
python -m vode view trace.json
```

### 5.3 高级选项

```bash
# 指定服务器地址和端口
python -m vode view trace.json --web --host 0.0.0.0 --port 8080

# 不自动打开浏览器
python -m vode view trace.json --web --no-browser
```

## 6. API 端点详情

### 6.1 GET /api/graph

获取完整的追踪图数据。

**响应示例**:

```json
{
  "version": "1.0",
  "timestamp": "2026-03-07T15:30:00.000000+00:00",
  "graph": {
    "root_call_ids": ["call_0"],
    "function_calls": [...],
    "variables": [...],
    "edges": [...]
  }
}
```

### 6.2 GET /api/node/{node_id}

获取指定节点的详细信息。

**参数**:

- `node_id`: 函数调用节点 ID

### 6.3 GET /api/search

搜索函数调用节点。

**查询参数**:

- `q`: 搜索关键词
- `limit`: 结果数量限制（默认 50）

### 6.4 GET /api/stats

获取图统计信息。

**响应示例**:

```json
{
  "function_count": 10,
  "variable_count": 25,
  "edge_count": 15,
  "max_depth": 4
}
```

## 7. 技术实现细节

### 7.1 架构设计

- **前后端分离**: FastAPI 提供 API，React 提供 UI
- **单页应用**: 所有交互在一个页面内完成
- **本地部署**: 服务器和客户端都在本地运行

### 7.2 数据流

1. CLI 命令启动 FastAPI 服务器
2. 服务器加载并解析 [`trace.json`](vode/src/vode/trace/serializer.py:54)
3. 服务器托管前端静态文件
4. 前端通过 REST API 获取数据
5. 前端使用 D3.js 和 Cytoscape.js 渲染可视化

### 7.3 可视化实现

- **调用树**: D3.js 层次布局，支持折叠/展开
- **数据流图**: Cytoscape.js 有向图布局，支持缩放/平移
- **值检查器**: Ant Design 面板组件

## 8. 已知限制

### 8.1 功能限制

- ❌ **导出功能未实现**: 暂不支持导出为 PNG/SVG（计划在后续版本）
- ❌ **实时追踪**: 仅支持查看已保存的追踪文件
- ❌ **编辑功能**: 只读查看器，不支持修改数据
- ❌ **性能分析**: 不提供性能剖析功能（Stage 3）
- ❌ **多文件对比**: 不支持同时查看多个追踪文件

### 8.2 性能限制

- ⚠️ **大型图性能**: 超过 1000 个节点可能有性能问题
- ⚠️ **浏览器兼容性**: 仅支持桌面浏览器（Chrome、Firefox、Safari、Edge）
- ⚠️ **移动端**: 不支持移动端和平板端

### 8.3 使用限制

- 需要先构建前端才能使用 Web 查看器
- 需要 Node.js 和 npm 来构建前端
- 服务器默认绑定到 `127.0.0.1`，仅本地访问

## 9. 测试覆盖

### 9.1 集成测试

- ✅ 追踪文件生成
- ✅ JSON 加载和解析
- ✅ API 端点响应
- ✅ 服务器启动和关闭

测试文件: [`vode/tests/test_stage2_integration.py`](vode/tests/test_stage2_integration.py)

### 9.2 手动测试

- ✅ Web 界面加载
- ✅ 调用树交互
- ✅ 数据流图交互
- ✅ 搜索功能
- ✅ 值检查器显示

## 10. 下一步计划（Stage 3）

### 10.1 性能优化

- 虚拟滚动和懒加载
- 大型图分页加载
- 图渲染性能优化

### 10.2 功能增强

- 导出为 PNG/SVG
- 高级搜索和过滤
- 自定义布局算法
- 键盘快捷键

### 10.3 性能分析

- 函数执行时间统计
- 火焰图可视化
- 内存使用分析
- 性能瓶颈识别

### 10.4 用户体验

- 移动端支持
- 主题切换
- 自定义配色
- 导出配置保存

## 11. 依赖项

### 11.1 Python 依赖

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
```

### 11.2 前端依赖

```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "antd": "^5.11.0",
  "d3": "^7.8.5",
  "cytoscape": "^3.28.0",
  "cytoscape-dagre": "^2.5.0"
}
```

## 12. 故障排除

### 12.1 前端未构建

**问题**: 访问 Web 界面显示 "Frontend not built yet"

**解决方案**:

```bash
cd vode/src/vode/view/frontend
npm install
npm run build
```

### 12.2 端口被占用

**问题**: 服务器启动失败，端口 8000 被占用

**解决方案**:

```bash
python -m vode view trace.json --web --port 8080
```

### 12.3 追踪文件未找到

**问题**: "Trace file not found"

**解决方案**: 确保追踪文件路径正确

```bash
python -m vode trace example.py -o trace.json
python -m vode view trace.json --web
```

## 13. 贡献指南

欢迎贡献！请参考以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 运行测试: `pytest tests/`
5. 提交 Pull Request

## 14. 许可证

MIT License

## 15. 致谢

感谢以下开源项目：

- React 和 TypeScript 社区
- FastAPI 和 Uvicorn
- D3.js 和 Cytoscape.js
- Ant Design 团队
