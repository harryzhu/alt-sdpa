import os
import time
import math
import numpy as np
# mlx.core 必须在 alt_sdpa 之前 import
import mlx.core as mx
from alt_sdpa import *


# 测试矩阵计算注意力
# 模拟 transformer 中：
# batch_size为20，8个头，256维（所以head_dim为32）的英译中句子，句子长度为43
#
# [B, H, L, Dh]
B, H, L, Dh = 20, 8, 43, 32

query = np.random.randn(B, H, L, Dh).astype(np.float32)
key = np.random.randn(B, H, L, Dh).astype(np.float32)
val = np.random.randn(B, H, L, Dh).astype(np.float32)
mask = np.random.randn(B, 1, L, L).astype(np.float32)

# 3种计算注意力的方式，结果一样
gpu_attention = alt_sdpa_gpu(query, key, val, mask)
cpu_attention = alt_sdpa_cpu(query, key, val, mask)
mlx_attention = alt_sdpa_mlx(query, key, val, mask)

# 由于计算机的浮点数是有误差的，所以结果存在极小误差是正常的
# 这个误差一般仅体现在小数点7位或8为之后
print(f"将 mlx_attention 最终结果作为标准值：")
print(f"gpu结果误差: {np.max(np.abs(gpu_attention - mlx_attention)):.9f}")
print(f"cpu结果误差: {np.max(np.abs(cpu_attention - mlx_attention)):.9f}")
