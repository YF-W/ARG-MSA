import torch
import numpy as np
import random
import logging
from torch.nn.parameter import UninitializedParameter

logger = logging.getLogger('MSA')

# pynvml is optional; fall back gracefully if not installed
try:
    import pynvml  # type: ignore
    _has_pynvml = True
except Exception:
    pynvml = None
    _has_pynvml = False


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _pick_gpu_with_nvml(memory_limit=1e16):
    pynvml.nvmlInit()
    try:
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        return dst_gpu_id, min_mem_used
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():
        # find most free gpu
        if _has_pynvml:
            try:
                dst_gpu_id, min_mem_used = _pick_gpu_with_nvml(memory_limit)
                logger.info(f'Found gpu {dst_gpu_id}, used memory {min_mem_used}.')
                gpu_ids.append(dst_gpu_id)
            except Exception as ex:
                # Some clusters ship pynvml in the env but do not expose the NVML
                # shared library inside the job container. Do not hard fail here.
                logger.warning(
                    "pynvml is installed but NVML is unavailable (%s). "
                    "Falling back to GPU 0 if CUDA is available.",
                    ex,
                )
                gpu_ids.append(0)
        else:
            # Fallback: pick GPU 0 if available
            logger.warning("pynvml not installed; defaulting to GPU 0 (if available).")
            gpu_ids.append(0)
    # device
    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()
    # logger.info("Let's use %d GPUs!" % len(gpu_ids))
    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device

def count_parameters(model):
    res = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, UninitializedParameter):
            # Lazy 模块尚未 materialize，先跳过
            continue
        res += p.numel()
    return res
