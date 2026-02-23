# DiffVG 完整指南

本仓库是一份 **DiffVG**（可微分矢量图形）的中文使用指南，涵盖安装配置、核心概念与使用样例，帮助你快速上手 DiffVG 并用于矢量图形优化、图像到矢量转换等任务。

## 什么是 DiffVG？

DiffVG 是一个可微分矢量图形渲染库，支持将矢量图形（SVG）渲染为光栅图像，并通过梯度反向传播对图形参数进行优化。应用于：

- 图像矢量化和风格化
- 矢量设计与优化

---

## 文档导航

| 文档 | 内容简介 |
|------|----------|
| [📦 安装指南](diffvg_install_guide.md) | Linux / Windows 下 DiffVG 的安装步骤、环境配置与常见问题 |
| [📐 图元与概念](diffvg_primitives_guide.md) | 五大图元（Circle、Ellipse、Path、Polygon、Rectangle）、ShapeGroup、颜色类型与渲染流程 |
| [🎨 实战：Collage Demo](collage_demo.md) | 以 `simple_collage` 为例，从轮廓提取到布局优化的完整 DiffVG 应用流程 |

### 快速跳转

- **刚入门？** → 从 [安装指南](diffvg_install_guide.md) 开始，按步骤完成环境搭建
- **准备开发？** → 阅读 [图元与概念](diffvg_primitives_guide.md)，了解 Circle、Ellipse、Path、Polygon、Rectangle 及 ShapeGroup 的用法
- **想看实战？** → 查看 [Collage Demo](collage_demo.md)，了解轮廓初始化、前向渲染、损失设计与最终 SVG 输出的完整流程 

---


## 参考链接

- [DiffVG 官方仓库](https://github.com/BachiLi/diffvg)
- [官方文档](https://deepwiki.com/BachiLi/diffvg)

---
 
