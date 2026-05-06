### 关于导入库的顺序问题
报错：OSError: [WinError 1114] Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```python
# 这段代码能跑
import os
os.add_dll_directory(...)
import torch    # ← torch 是 add_dll_directory 之后第一个重型导入
print("成功")

# ... 后面再导入 pandas 等其他库就没事

# ==========================================

# 这段代码会报错 
import os
os.add_dll_directory(...)
import pandas as pd # ← pandas 成了第一个重型导入
import torch        # ← torch 在 pandas 之后导入，结果加载 DLL 失败
```
原因：
PyTorch 的 CUDA DLL 文件（c10.dll 及其依赖）所在的目录，不在 Windows 默认的 DLL 搜索路径中。

环境里的 pandas 被 d2l 包依赖。import pandas 的过程可能触发 Python 对整个 site-packages 的扫描，进而提前、或在不恰当的时机加载了 d2l 中依赖 torch 的模块，等轮到 import torch 时，DLL 搜索上下文已经被污染了


### Windows 下 PyTorch GRU/LSTM 内部 dropout 的栈溢出问题
在 GPU 上运行带有 dropout > 0 的 nn.GRU 时，PyTorch 默认会调用 NVIDIA 的 cuDNN 加速库。cuDNN 对 RNN 的 dropout 进行了特殊优化，但其内部实现使用了深度递归函数来生成 dropout 掩码（或处理变长序列的 dropout 状态）

Windows 下每个线程的默认栈大小通常为 1 MB。
Linux 下通常为 8 MB。
cuDNN 的递归深度与 num_layers、序列长度、dropout 概率等因素相关。即使 num_layers=2，也可能触发对每一时间步、每一层的递归调用，导致栈内存迅速耗尽，从而发生 栈缓冲区溢出（stack buffer overflow），表现为退出代码 0xC0000409

解决办法：
在 GRU 外部 加入 nn.Dropout，而不是使用 GRU 内置的 dropout。例如：
```python
self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=0)   # 内部 dropout 设为 0
self.dropout = nn.Dropout(dropout)   # 独立 dropout 层
```