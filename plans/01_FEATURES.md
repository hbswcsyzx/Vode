# Vode 功能介绍

## 概述

**Vode** (View Your Code) 是一个 Python 代码可视化与分析工具，专为理解复杂程序执行流程而设计，特别适用于 PyTorch 深度学习模型的分析。

Vode 的核心理念是将任意 Python 程序建模为 **Input → Op → Output** 的递归结构，并提供交互式的层级展开能力，让开发者能够从宏观到微观逐层深入理解代码执行。

---

## 核心功能

### 1. 双模式捕获

#### 1.1 函数流捕获 (Function Flow)

**目标**：捕获程序从 `main()` 到最底层的完整函数调用栈。

**适用场景**：

- 理解程序整体执行流程
- 追踪函数调用关系
- 定位性能瓶颈
- 调试复杂的函数嵌套

**捕获内容**：

- 所有用户定义的函数
- 类方法调用
- PyTorch 模块的 forward 方法(如 `Linear.forward()`, `Conv2d.forward()`)
- 函数调用的层级关系和顺序

**可选数据捕获**：

- 函数参数的名称、类型、值
- 返回值
- 执行时间
- 调用次数

**示例**：

```python
# 你的代码
def preprocess(data):
    return normalize(data)

def train_step(model, data):
    data = preprocess(data)
    output = model(data)
    loss = compute_loss(output)
    return loss

# 使用 Vode 捕获
$ vode trace --mode function train.py
```

**捕获结果**：

```
main()
├── train_step()
│   ├── preprocess()
│   │   └── normalize()
│   ├── model.__call__()
│   │   ├── Linear.forward()
│   │   ├── ReLU.forward()
│   │   └── Linear.forward()
│   └── compute_loss()
```

---

#### 1.2 计算流捕获 (Computation Flow)

**目标**：只捕获实际的计算图，过滤掉 Python 函数包装，专注于数据流动。

**适用场景**：

- 理解模型架构
- 分析数据流向
- 检查张量形状变化
- 优化模型结构

**捕获内容**：

- PyTorch 模块实例(`nn.Linear`, `nn.Conv2d`, `nn.ReLU` 等)
- 模块之间的数据流连接
- 张量的形状、类型、设备信息

**可选数据捕获**：

- 输入/输出张量的 shape, dtype, device
- 张量统计信息(min, max, mean, std)
- 模块参数数量
- 内存占用

**示例**：

```python
# 你的模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)

# 使用 Vode 捕获
$ vode trace --mode computation --capture-data train.py
```

**捕获结果**：

```
Input [1, 784]
  ↓
Linear(784→256) [1, 256]
  ↓
ReLU [1, 256]
  ↓
Dropout(p=0.5) [1, 256]
  ↓
Linear(256→10) [1, 10]
  ↓
Output [1, 10]
```

---

### 2. 双模式展示

#### 2.1 静态导出 (Static Export)

**目标**：生成固定深度的可视化图片，用于文档和演示。

**使用方式**：

```bash
# 导出函数流图(深度3层)
$ vode export trace.json output.png --mode function --depth 3

# 导出计算流图(深度5层，SVG格式)
$ vode export trace.json model.svg --mode computation --depth 5
```

**特点**：

- 支持 PNG, SVG, PDF 格式
- 用户指定展示深度
- 清晰的层级结构
- 适合放入论文、报告、文档

**展示内容**：

- **函数流**：函数名、文件位置、调用次数
- **计算流**：模块类型、张量形状、参数数量

---

#### 2.2 交互式查看 (Interactive View)

**目标**：提供可交互的图形界面，支持动态展开/折叠节点，查看详细信息。

**使用方式**：

```bash
# 打开交互式查看器
$ vode view trace.json

# 自动在浏览器或 VSCode 中打开可视化面板
```

**核心交互功能**：

##### a) 节点展开/折叠

- **单击节点**：展开到下一层级
- **双击节点**：折叠回上一层级
- **右键菜单**：展开所有子节点、折叠所有子节点
- **键盘快捷键**：
  - `E` - 展开选中节点
  - `C` - 折叠选中节点
  - `A` - 展开所有
  - `Shift+C` - 折叠所有

##### b) 节点详情面板

点击节点后，右侧面板显示详细信息：

**函数流节点**：

```
函数名：train_step
文件：train.py:45
调用次数：100
总执行时间：2.34s
平均执行时间：23.4ms

参数：
  - model: <class 'torch.nn.Module'>
  - data: torch.Tensor [32, 784]
    ├─ dtype: float32
    ├─ device: cuda:0
    ├─ min: -2.1, max: 3.4
    └─ mean: 0.02, std: 1.01

返回值：
  - loss: torch.Tensor [1]
    └─ value: 0.456
```

**计算流节点**：

```
模块类型：Linear
名称：model.fc1
参数数量：200,960 (全部可训练)

输入：
  - Tensor [32, 784]
    ├─ dtype: float32
    ├─ device: cuda:0
    ├─ min: -2.1, max: 3.4
    └─ mean: 0.02, std: 1.01

输出：
  - Tensor [32, 256]
    ├─ dtype: float32
    ├─ device: cuda:0
    ├─ min: -1.8, max: 2.9
    └─ mean: 0.01, std: 0.95

模块配置：
  - in_features: 784
  - out_features: 256
  - bias: True
```

##### c) 搜索与过滤

- **搜索框**：按名称搜索节点
- **类型过滤**：只显示特定类型的节点(如只显示 Conv2d)
- **深度过滤**：只显示特定深度范围的节点
- **性能过滤**：只显示执行时间超过阈值的节点

##### d) 图形操作

- **缩放**：鼠标滚轮或触控板手势
- **平移**：拖拽画布
- **自动布局**：一键重新排列节点
- **适应窗口**：自动缩放以显示全部内容
- **小地图**：显示整体结构，快速导航

##### e) 导出功能

- **导出当前视图**：将当前展开状态导出为图片
- **导出子图**：只导出选中节点及其子树
- **导出数据**：将节点详情导出为 JSON/CSV

---

### 3. 面板式开发 (Visual Programming) - 未来功能

**目标**：像 Scratch 一样，通过拖拽和连线的方式搭建可运行的模型。

**使用场景**：

- 快速原型设计
- 教学演示
- 模型架构探索
- 无代码/低代码开发

**功能设计**：

#### 3.1 模块库面板(左侧)

```
📦 基础层
  ├─ Linear
  ├─ Conv2d
  ├─ BatchNorm2d
  └─ ...

📦 激活函数
  ├─ ReLU
  ├─ Sigmoid
  ├─ Tanh
  └─ ...

📦 池化层
  ├─ MaxPool2d
  ├─ AvgPool2d
  └─ ...

📦 自定义模块
  ├─ 我的模块1
  └─ 我的模块2
```

#### 3.2 画布(中间)

- 从左侧拖拽模块到画布
- 通过连线连接模块的输入输出
- 自动检查形状兼容性
- 实时显示数据流形状

#### 3.3 属性面板(右侧)

选中模块后，配置其参数：

```
Linear 层配置
├─ in_features: 784
├─ out_features: 256
├─ bias: ✓
└─ device: cuda:0
```

#### 3.4 代码生成

点击"生成代码"按钮，自动生成 PyTorch 代码：

```python
import torch.nn as nn

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

#### 3.5 双向同步

- **代码 → 图形**：导入现有代码，自动生成可视化图形
- **图形 → 代码**：修改图形，实时更新代码
- **实时预览**：在画布上直接运行模型，查看中间结果

---

## 完整工作流

### 场景1：理解现有代码

```bash
# 1. 捕获函数流
$ vode trace --mode function --capture-data my_script.py

# 2. 交互式查看
$ vode view my_script.vode.json

# 3. 在浏览器中：
#    - 从 main() 开始逐层展开
#    - 点击感兴趣的函数查看参数和返回值
#    - 搜索特定函数名
#    - 导出关键路径的子图
```

### 场景2：分析模型架构

```bash
# 1. 捕获计算流
$ vode trace --mode computation --capture-data train_model.py

# 2. 导出静态图(用于文档)
$ vode export train_model.vode.json model_arch.svg --mode computation --depth 10

# 3. 交互式查看(用于分析)
$ vode view train_model.vode.json
#    - 检查每层的输入输出形状
#    - 查看参数数量分布
#    - 识别瓶颈层
```

### 场景3：性能分析

```bash
# 1. 捕获函数流(带数据)
$ vode trace --mode function --capture-data slow_script.py

# 2. 交互式查看
$ vode view slow_script.vode.json

# 3. 在浏览器中：
#    - 按执行时间排序节点
#    - 过滤出耗时超过100ms的函数
#    - 展开慢函数，查看其子调用
#    - 定位性能瓶颈
```

### 场景4：教学演示

```bash
# 1. 捕获简单模型
$ vode trace --mode computation examples/simple_cnn.py

# 2. 导出多层深度的图片
$ vode export simple_cnn.vode.json cnn_depth1.png --depth 1
$ vode export simple_cnn.vode.json cnn_depth2.png --depth 2
$ vode export simple_cnn.vode.json cnn_depth3.png --depth 3

# 3. 在PPT中逐步展示模型结构
```

### 场景5：快速原型(未来)

```bash
# 1. 启动可视化编辑器
$ vode editor

# 2. 在浏览器中：
#    - 拖拽模块搭建模型
#    - 连线并配置参数
#    - 点击"生成代码"
#    - 保存为 model.py

# 3. 直接使用生成的代码
$ python train.py --model model.py
```

---

## 命令行接口

### `vode trace` - 捕获执行轨迹

```bash
vode trace [OPTIONS] SCRIPT [ARGS...]

选项：
  --mode {function|computation}  捕获模式(必选)
  --capture-data                 捕获详细数据(参数、张量统计等)
  --output, -o PATH              输出文件路径(默认：SCRIPT.vode.json)
  --max-depth INT                最大捕获深度
  --filter PATTERN               过滤规则(正则表达式)

示例：
  vode trace --mode function train.py
  vode trace --mode computation --capture-data model.py --output model_trace.json
  vode trace --mode function --max-depth 5 deep_recursion.py
```

### `vode export` - 导出静态图片

```bash
vode export [OPTIONS] TRACE_FILE OUTPUT_FILE

选项：
  --mode {function|computation}  展示模式(必选)
  --depth INT                    展示深度(默认：3)
  --format {png|svg|pdf}         输出格式(从文件扩展名推断)
  --theme {light|dark}           主题
  --layout {TB|LR|BT|RL}         布局方向(上下/左右/下上/右左)

示例：
  vode export trace.json output.png --mode function --depth 3
  vode export trace.json model.svg --mode computation --depth 5 --layout LR
```

### `vode view` - 交互式查看

```bash
vode view [OPTIONS] TRACE_FILE

选项：
  --port INT                     服务器端口(默认：8000)
  --no-browser                   不自动打开浏览器
  --vscode                       在 VSCode webview 中打开

示例：
  vode view trace.json
  vode view trace.json --port 8080
  vode view trace.json --vscode
```

### `vode editor` - 可视化编辑器(未来)

```bash
vode editor [OPTIONS] [MODEL_FILE]

选项：
  --port INT                     服务器端口(默认：8000)
  --template {resnet|vgg|...}    从模板开始

示例：
  vode editor
  vode editor my_model.py
  vode editor --template resnet18
```

---

## Python API

除了命令行，Vode 也提供 Python API：

```python
from vode import FunctionTracer, ComputationTracer, Visualizer

# 函数流捕获
tracer = FunctionTracer(capture_data=True)
tracer.start()
my_function()
tracer.stop()
tracer.save('trace.json')

# 计算流捕获
tracer = ComputationTracer(capture_data=True)
output = tracer.trace(model, input_tensor)
tracer.save('model_trace.json')

# 可视化
viz = Visualizer.from_file('trace.json')
viz.export('output.png', mode='function', depth=3)
viz.serve(port=8000)  # 启动交互式查看器
```

---

## 与现有工具的对比

| 工具               | 函数流 | 计算流 | 交互式         | 数据捕获  | 可视化编辑 |
|--------------------|--------|--------|----------------|-----------|------------|
| **torchview**      | ❌      | ✅      | ❌              | ✅(仅形状) | ❌          |
| **torch.profiler** | ✅      | ❌      | ✅(TensorBoard) | ✅(性能)   | ❌          |
| **py-spy**         | ✅      | ❌      | ❌              | ❌         | ❌          |
| **snakeviz**       | ✅      | ❌      | ✅              | ✅(性能)   | ❌          |
| **Netron**         | ❌      | ✅      | ✅              | ✅(静态)   | ❌          |
| **Vode**           | ✅      | ✅      | ✅              | ✅(完整)   | ✅(未来)    |

**Vode 的独特优势**：

- 同时支持函数级和计算级视图
- 递归展开的交互式探索
- 丰富的数据捕获(不仅是形状，还有值、统计信息)
- 未来支持可视化编程

---

## 适用人群

- **深度学习研究者**：理解复杂模型架构，调试训练流程
- **工程师**：分析性能瓶颈，优化代码结构
- **教师/学生**：教学演示，学习理解
- **开源贡献者**：快速理解陌生代码库
- **算法工程师**：原型设计，快速迭代

---

## 安装

```bash
# 从 PyPI 安装
pip install vode

# 从源码安装
git clone https://github.com/your-org/vode.git
cd vode
pip install -e .
```

---

## 总结

Vode 是一个强大的代码可视化工具，通过双模式捕获(函数流/计算流)和双模式展示(静态导出/交互式查看)，帮助开发者从不同角度理解代码执行。未来还将支持可视化编程，让模型开发更加直观高效。
