# Stage 2 快速入门指南

## 1. 安装

### 1.1 系统要求

- Python 3.10+
- Node.js 16+ 和 npm（用于构建前端）
- 桌面浏览器（Chrome、Firefox、Safari 或 Edge）

### 1.2 安装 Vode

```bash
cd vode
pip install -e .
```

### 1.3 构建前端（首次使用必需）

```bash
cd src/vode/view/frontend
npm install
npm run build
cd ../../../..
```

**注意**: 前端只需构建一次，除非前端代码有更新。

## 2. 基本使用

### 2.1 生成追踪文件

创建一个简单的 Python 脚本 `example.py`:

```python
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def compute(x, y):
    sum_val = add(x, y)
    prod_val = multiply(x, y)
    return sum_val, prod_val

if __name__ == '__main__':
    result = compute(3, 4)
    print(f"Result: {result}")
```

生成追踪文件:

```bash
python -m vode trace example.py -o trace.json
```

### 2.2 使用 Web 查看器

```bash
python -m vode view trace.json --web
```

浏览器会自动打开 `http://127.0.0.1:8000`。

### 2.3 使用文本查看器（默认）

```bash
python -m vode view trace.json
```

## 3. Web 界面使用

### 3.1 界面布局

Web 界面包含以下区域：

- **顶部栏**: 文件名、搜索框、统计信息
- **左侧面板**: 视图切换、函数列表、过滤选项
- **主视图区**: 调用树或数据流图
- **底部面板**: 值检查器（显示选中节点详情）

### 3.2 调用树视图

**功能**:

- 查看函数调用层级关系
- 展开/折叠节点
- 点击节点查看详情
- 显示参数和返回值预览

**操作**:

- 单击节点：选中并在值检查器中显示详情
- 双击节点：展开/折叠子节点
- 滚动：浏览整个调用树

### 3.3 数据流图视图

**功能**:

- 查看函数间数据传递关系
- 识别数据生产者和消费者
- 追踪对象流向

**操作**:

- 单击节点：选中并显示详情
- 滚轮：缩放图形
- 拖动：平移视图
- 工具栏按钮：重置视图、适配屏幕

### 3.4 值检查器

选中任意函数调用节点后，值检查器会显示：

- **参数列表**: 函数输入参数及其值
- **返回值**: 函数返回值
- **位置信息**: 源文件路径和行号
- **Tensor 信息**: 如果是 PyTorch tensor，显示 shape、dtype、device 等

### 3.5 搜索功能

在顶部搜索框中输入关键词，可以搜索：

- 函数名
- 限定名（包含模块路径）
- 文件路径

搜索结果会在左侧面板显示，点击可快速定位。

## 4. 高级选项

### 4.1 自定义服务器配置

```bash
# 指定主机和端口
python -m vode view trace.json --web --host 0.0.0.0 --port 8080

# 不自动打开浏览器
python -m vode view trace.json --web --no-browser
```

### 4.2 追踪配置

生成追踪文件时可以配置：

```bash
# 限制追踪深度
python -m vode trace example.py --max-depth 5

# 排除特定模块
python -m vode trace example.py --exclude "torch.*" --exclude "numpy.*"

# 配置值捕获策略
python -m vode trace example.py --value-policy preview
```

值策略选项：

- `none`: 不捕获值
- `stats_only`: 仅捕获统计信息（默认）
- `preview`: 捕获预览文本
- `full`: 捕获完整值（可能很大）

## 5. 常见问题

### 5.1 前端未构建

**问题**: 访问 Web 界面显示 "Frontend not built yet"

**解决方案**:

```bash
cd vode/src/vode/view/frontend
npm install
npm run build
```

### 5.2 端口被占用

**问题**: `Address already in use`

**解决方案**: 使用不同端口

```bash
python -m vode view trace.json --web --port 8080
```

### 5.3 追踪文件过大

**问题**: 浏览器加载缓慢或卡顿

**解决方案**:

- 使用 `--max-depth` 限制追踪深度
- 使用 `--exclude` 排除不需要的模块
- 使用 `--value-policy none` 或 `stats_only` 减少数据量

### 5.4 找不到 vode 命令

**问题**: `command not found: vode`

**解决方案**: 使用完整命令

```bash
python -m vode trace example.py
python -m vode view trace.json --web
```

### 5.5 PyTorch tensor 信息不显示

**问题**: Tensor 只显示为普通对象

**解决方案**: 确保安装了 PyTorch

```bash
pip install torch
```

## 6. 性能建议

### 6.1 追踪大型程序

对于大型程序，建议：

1. **限制深度**: 使用 `--max-depth 10` 避免过深追踪
2. **排除库代码**: 使用 `--exclude` 排除第三方库
3. **减少值捕获**: 使用 `--value-policy stats_only`

示例:

```bash
python -m vode trace large_app.py \
  --max-depth 10 \
  --exclude "torch.*" \
  --exclude "numpy.*" \
  --exclude "pandas.*" \
  --value-policy stats_only \
  -o trace.json
```

### 6.2 查看大型追踪文件

如果追踪文件包含超过 1000 个节点：

- 使用搜索功能快速定位
- 使用过滤功能只显示相关部分
- 考虑重新追踪并限制范围

## 7. 下一步

- 查看 [`vode/docs/stage2/report.md`](vode/docs/stage2/report.md) 了解完整功能
- 查看 [`vode/docs/stage2/capabilities.md`](vode/docs/stage2/capabilities.md) 了解能力边界
- 查看 [`vode/docs/stage2/design.md`](vode/docs/stage2/design.md) 了解设计细节

## 8. 获取帮助

```bash
# 查看帮助
python -m vode --help
python -m vode trace --help
python -m vode view --help
```

## 9. 示例工作流

完整的使用流程：

```bash
# 1. 创建测试脚本
cat > test.py << 'EOF'
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"5! = {result}")
EOF

# 2. 生成追踪
python -m vode trace test.py -o trace.json

# 3. 查看追踪（Web）
python -m vode view trace.json --web

# 4. 或查看追踪（文本）
python -m vode view trace.json
```

## 10. 故障排除清单

遇到问题时，按顺序检查：

1. ✅ Python 版本是否 >= 3.10
2. ✅ 是否已安装 Vode: `pip list | grep vode`
3. ✅ 前端是否已构建: `ls vode/src/vode/view/frontend/dist`
4. ✅ 追踪文件是否存在: `ls trace.json`
5. ✅ 端口是否可用: `lsof -i :8000` (macOS/Linux)
6. ✅ 浏览器控制台是否有错误

如果问题仍未解决，请提交 issue 并附上：

- 错误信息
- Python 版本
- 操作系统
- 追踪文件大小
- 使用的命令
