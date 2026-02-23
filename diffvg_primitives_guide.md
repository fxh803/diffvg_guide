# DiffVG 可微分矢量图元详解

本文档旨在详细说明 diffvg 库所支持的核心矢量图形图元。这些图元是构建可微分渲染场景的基本单元，其参数均可通过梯度进行优化。

## 1. 核心图元概览

diffvg 定义了五种基本的矢量图形图元，分别是：

- **Circle** (圆形)
- **Ellipse** (椭圆形)
- **Path** (路径)
- **Polygon** (多边形)
- **Rectangle** (矩形)

所有图元都共享一些通用属性，如 `stroke_width` (描边宽度) 和可选的 `id` (标识符)。它们必须被组织在 **ShapeGroup** (图元组) 中，以定义其填充色、描边色和空间变换。

## 2. 图元类型详述

### 2.1 Circle (圆形)

圆形由圆心和半径唯一确定。

**必需参数：**

- `radius`: 标量，表示圆的半径。
- `center`: 二维张量 `[x, y]`，表示圆心坐标。

**可选参数：**

- `stroke_width`: 描边宽度（默认 1.0）。
- `id`: 可选标识符字符串。

**示例 (PyTorch)：**

```python
import torch

circle = Circle(
    radius=torch.tensor(5.0),
    center=torch.tensor([100.0, 100.0]),
    stroke_width=torch.tensor(2.0)
)
```

### 2.2 Ellipse (椭圆形)

椭圆形由中心和两个轴向的半径定义。

**必需参数：**

- `radius`: 二维张量 `[x_radius, y_radius]`，分别表示 X 轴和 Y 轴方向的半径。
- `center`: 二维张量 `[x, y]`，表示椭圆中心坐标。

**可选参数：**

- `stroke_width`: 描边宽度（默认 1.0）。
- `id`: 可选标识符字符串。

**示例 (PyTorch)：**

```python
ellipse = Ellipse(
    radius=torch.tensor([10.0, 5.0]),
    center=torch.tensor([100.0, 100.0]),
    stroke_width=torch.tensor(2.0)
)
```

### 2.3 Path (贝塞尔路径)

路径是最灵活的图元，用于定义由控制点构成的贝塞尔曲线，可表现复杂形状。

**必需参数：**

- `num_control_points`: 张量，指定每个贝塞尔线段所需的控制点数量。
- `points`: 控制点坐标张量，形状为 `[num_points, 2]`。
- `is_closed`: 布尔值，指示路径是否闭合。

**可选参数：**

- `stroke_width`: 描边宽度（默认 1.0）。
- `id`: 可选标识符字符串。
- `use_distance_approx`: 布尔值，指示是否使用距离场近似。

**示例 (PyTorch)：**

```python
path = Path(
    num_control_points=torch.tensor([3, 3]),  # 两个三次贝塞尔线段
    points=torch.tensor([[0.0, 0.0], [10.0, 30.0], [20.0, 0.0],
                         [20.0, 0.0], [30.0, 30.0], [40.0, 0.0]]),
    is_closed=False,
    stroke_width=torch.tensor(2.0)
)
```

### 2.4 Polygon (多边形)

多边形由一系列顶点定义。

**必需参数：**

- `points`: 顶点坐标张量，形状为 `[num_points, 2]`。
- `is_closed`: 布尔值，指示多边形是否闭合（通常为 `True`）。

**可选参数：**

- `stroke_width`: 描边宽度（默认 1.0）。
- `id`: 可选标识符字符串。

**示例 (PyTorch)：**

```python
polygon = Polygon(
    points=torch.tensor([[0.0, 0.0], [100.0, 0.0],
                         [100.0, 100.0], [0.0, 100.0]]),
    is_closed=True,
    stroke_width=torch.tensor(2.0)
)
```

### 2.5 Rectangle (矩形)

矩形由最小角和最大角坐标定义的轴对齐矩形。

**必需参数：**

- `p_min`: 二维张量 `[x, y]`，表示矩形左上角坐标。
- `p_max`: 二维张量 `[x, y]`，表示矩形右下角坐标。

**可选参数：**

- `stroke_width`: 描边宽度（默认 1.0）。
- `id`: 可选标识符字符串。

**示例 (PyTorch)：**

```python
rect = Rect(
    p_min=torch.tensor([0.0, 0.0]),
    p_max=torch.tensor([100.0, 50.0]),
    stroke_width=torch.tensor(2.0)
)
```

## 3. ShapeGroup (图元组)

图元本身不直接持有颜色或复杂变换属性。这些属性由 **ShapeGroup** 统一管理。一个图元组将多个图元聚合在一起，并为其定义共享的外观属性。

**核心功能：**

1. **聚合图元**：通过 `shape_ids` 引用场景中的图元列表。
2. **定义外观**：设置 `fill_color` (填充色) 和 `stroke_color` (描边色)。
3. **施加变换**：通过 `shape_to_canvas` (一个 3×3 变换矩阵) 对整个图元组进行平移、旋转、缩放等操作。

**参数：**

- `shape_ids`: 属于此组的图元索引张量。
- `fill_color`: 填充颜色，可以是常量色、线性渐变或径向渐变。
- `stroke_color`: 描边颜色，类型同 `fill_color`，可为 null。
- `shape_to_canvas`: 3×3 变换矩阵。
- `use_even_odd_rule`: 布尔值，确定填充的奇偶规则。
- `id`: 可选标识符字符串。

**示例 (PyTorch)：**

```python
shape_group = ShapeGroup(
    shape_ids=torch.tensor([0, 1, 2]),  # 引用场景中的前三个图元
    fill_color=torch.tensor([1.0, 0.5, 0.2, 1.0]),  # RGBA，橙色
    use_even_odd_rule=True,
    stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),  # RGBA，黑色描边
    shape_to_canvas=torch.eye(3)  # 单位矩阵，即无变换
)
```

## 4. 颜色类型

diffvg 支持三种颜色类型，用于 `fill_color` 和 `stroke_color`：

1. **Constant Color (常量颜色)**：单一的 RGBA 颜色值。

2. **Linear Gradient (线性渐变)**：沿一条直线插值的渐变。
   - 参数：`begin` (起点)、`end` (终点)、`offsets` (颜色停点)、`stop_colors` (各停点颜色)。

3. **Radial Gradient (径向渐变)**：从一个中心点向外径向插值的渐变。
   - 参数：`center` (中心点)、`radius` (半径)、`offsets` (颜色停点)、`stop_colors` (各停点颜色)。

## 5. 与渲染系统的集成

在渲染前，所有高级图元对象都会被 `serialize_scene` 函数序列化成线性参数列表，以供底层的 C++ 核心渲染器高效处理。渲染系统还支持多种抗锯齿滤波器（如 box、tent、hann）和输出类型（彩色图像或 Signed Distance Field）。

## 6. 图元属性速查表

| 图元 | 必需参数 | 可选参数 |
|------|----------|----------|
| Circle | radius, center | stroke_width, id |
| Ellipse | radius (x, y), center | stroke_width, id |
| Path | num_control_points, points, is_closed | stroke_width, id, use_distance_approx |
| Polygon | points, is_closed | stroke_width, id |
| Rectangle | p_min, p_max | stroke_width, id |

**说明**：所有图元的填充和描边渲染都通过其所属的 ShapeGroup 控制。每个图元都可以通过其图元组的 `shape_to_canvas` 矩阵进行几何变换。

---

*本文档内容参考[https://deepwiki.com/BachiLi/diffvg/4-vector-graphics-primitives](https://deepwiki.com/BachiLi/diffvg/4-vector-graphics-primitives)*
