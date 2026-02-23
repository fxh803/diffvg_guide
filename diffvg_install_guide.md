# DiffVG 使用指南

DiffVG 是一个可微分矢量图形渲染库，支持将矢量图形（SVG）渲染为光栅图像，并可通过梯度进行优化。本文档介绍其安装与基本使用。

---

## 安装

### 1. 克隆仓库（Linux / Windows 通用）

```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
```


完成上述步骤后，根据你的操作系统选择下方对应小节继续安装。

---

### 2. Linux 下安装

**创建 Python 环境**

```bash
conda create -n diffvg python=3.10 （推荐版本）
conda activate diffvg
```

**CUDA Toolkit**

先运行 `nvidia-smi` 确认驱动支持的最高 CUDA 版本，再下载安装不超过该版本的 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)。

输出样例（关注 **CUDA Version** 一行的数值）：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 551.23       Driver Version: 551.23       CUDA Version: 12.4    |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... On   | 00000000:01:00.0  On |                  N/A |
| 41%   41C    P8     4W /  50W |    699MiB /  4096MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+
```

此处 `CUDA Version: 12.4` 表示当前驱动最高支持 CUDA 12.4，应安装 12.4 或更低版本的 Toolkit。

安装后验证 CUDA：

```bash
nvcc --version
```
输出应如下，表示cuda安装成功
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
```

**安装 PyTorch**（使用 GPU，并对应cuda版本）

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
> cu124指cuda12.4，同理根据cuda版本自行进行更改
 

**编译并安装**

在项目根目录 `diffvg` 下执行：

```bash
python setup.py install
```

安装完成后，在 Python 中执行 `import diffvg` 验证是否成功。

---

### 3. Windows 下安装

在 Windows 下安装 Diffvg 较为复杂，请严格按照以下步骤及版本进行操作。

**环境要求**

- **Visual Studio 2022**：需安装「使用 C++ 的桌面开发」工作负载
- **CUDA Toolkit**：版本12.4。
- **Miniconda**：Miniconda基础python版本与虚拟环境均使用 **Python 3.10**（创建环境时指定 `python=3.10`）

**源码修改**（在编译前必须完成）

1. **diffvg.h**  
   找到第 97 行，将 `inline double log2{}` 注释掉：

   ```cpp
   // inline double log2{}
   ```

2. **setup.py**  
   找到第 39 行，将 `get_config_var('LIBDIR')` 改为：

   ```python
   get_config_var('LIBDEST')
   ```

**创建 Python 环境**

```bash
conda create -n diffvg python=3.10
conda activate diffvg
```

**安装 CUDA toolkit 与 PyTorch 依赖**

此步骤与 Linux 相同：先运行 `nvidia-smi` 确认驱动支持的 CUDA 版本 → 安装对应版本的 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 并用 `nvcc --version` 验证 → 再根据 CUDA 版本安装 PyTorch。具体命令见上方「2. Linux 下安装」中的 **CUDA Toolkit** 与 **安装 PyTorch** 小节。


**编译并安装**

在项目根目录 `diffvg` 下执行：

```bash
python setup.py install
```

安装完成后，在 Python 中执行 `import diffvg` 验证是否成功。

---

## 常见问题

**Windows 下执行 `import diffvg` 报错 `ModuleNotFoundError: No module named 'diffvg'`**

安装完成后验证时出现该错误，可尝试以下方法：将 环境目录（具体通过**conda info**查看）对应文件夹里的 `diffvg` 文件重命名为 `diffvg.pyd`。

路径形如：

```
C:\ProgramData\miniconda3\envs\xxx\Lib\site-packages\diffvg-0.0.1-py3.x-win-amd64.egg
```
或
```
C:\ProgramData\miniconda3\envs\xxx\Lib\site-packages 
```
把其中的无扩展名文件 `diffvg` 改名为 `diffvg.pyd`。完成后再次执行 `import diffvg` 验证。

---

  
