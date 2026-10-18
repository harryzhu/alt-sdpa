#
import numpy as np
import math
import mlx.core as mx
from .alt_sdpa_ext import sdpa_cpu, sdpa_gpu

def alt_sdpa_gpu(q,k,v,mask):
    _b, _h, _l, _dh = q.shape
    q = mx.array(q)
    kt = mx.array(k).transpose(0,1,3,2)
    v = mx.array(v)
    mask = mx.array(mask)
    out_atten = sdpa_gpu(q,kt,v,mask)
    gpu_attention = mx.array(out_atten).transpose(0,2,1,3).reshape(_b,_l,_h*_dh)
    return gpu_attention

def alt_sdpa_cpu(q,k,v,mask):
    _b,_h,_l,_dh = q.shape
    q = np.ascontiguousarray(q)
    kt = np.ascontiguousarray(k.transpose(0,1,3,2))
    v = np.ascontiguousarray(v)
    mask = np.ascontiguousarray(mask)
    scores = np.ascontiguousarray(np.empty(shape=[_b,_h,_l,_l], dtype=np.float32))
    out_atten = np.ascontiguousarray(np.empty(shape=[_b,_h,_l,_dh], dtype=np.float32))
    sdpa_cpu(q,kt,v,mask,scores,out_atten)
    cpu_attention = out_atten.transpose((0,2,1,3)).reshape((_b,_l,_h * _dh))
    return mx.array(cpu_attention)

def alt_sdpa_mlx(q,k,v,mask):
    _b,_h,_l,_dh = q.shape
    alpha = 1.0/math.sqrt(_dh)
    kt = mx.array(k).transpose(0,1,3,2)
    wt = mx.array((q @ kt) * alpha + mask)
    sftwt = mx.softmax(wt, axis=-1)
    out_atten = sftwt @ mx.array(v)
    mlx_attention = out_atten.transpose(0,2,1,3).reshape(_b,_l,_h*_dh)
    return mlx_attention
