import os
import yaml
import random
import torch
from torch import distributed as torch_dist
import torch.multiprocessing as mp
import torch.backends.cudnn
import numpy as np

import datetime
import subprocess
import platform
import warnings


def set_multi_processing(mp_start_method: str = 'fork',
                         opencv_num_threads: int = 0,
                         distributed: bool = False) -> None:
    """Set multi-processing related environment.

    Args:
        mp_start_method (str): Set the method which should be used to start
            child processes. Defaults to 'fork'.
        opencv_num_threads (int): Number of threads for opencv.
            Defaults to 0.
        distributed (bool): True if distributed environment.
            Defaults to False.
    """
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        current_method = mp.get_start_method(allow_none=True)
        if (current_method is not None and current_method != mp_start_method):
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`. You can '
                'change this behavior by changing `mp_start_method` in '
                'your config.')
        mp.set_start_method(mp_start_method, force=True)

    try:
        import cv2

        # disable opencv multithreading to avoid system being overloaded
        cv2.setNumThreads(opencv_num_threads)
    except ImportError:
        pass

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if 'OMP_NUM_THREADS' not in os.environ and distributed:
        omp_num_threads = 1
        warnings.warn(
            'Setting OMP_NUM_THREADS environment variable for each process'
            f' to be {omp_num_threads} in default, to avoid your system '
            'being overloaded, please further tune the variable for '
            'optimal performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and distributed:
        mkl_num_threads = 1
        warnings.warn(
            'Setting MKL_NUM_THREADS environment variable for each process'
            f' to be {mkl_num_threads} in default, to avoid your system '
            'being overloaded, please further tune the variable for '
            'optimal performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

def is_distributed():
    if not (torch_dist.is_available() and torch_dist.is_initialized()):
        return False
    return True


def distributed_rank():
    if not is_distributed():
        return 0
    else:
        return torch_dist.get_rank()


def is_main_process():
    return distributed_rank() == 0


def distributed_world_size():
    if is_distributed():
        return torch_dist.get_world_size()
    else:
        return 1


def set_seed(seed: int):
    seed = seed + distributed_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # If you don't want to wait until the universe is silent, do not use this below code :)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return


def yaml_to_dict(path: str):
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


def labels_to_one_hot(labels: np.ndarray, class_num: int):
    return np.eye(N=class_num)[labels]


def inverse_sigmoid(x, eps=1e-5):
    """
    if      x = 1/(1+exp(-y))
    then    y = ln(x/(1-x))
    Args:
        x:
        eps:

    Returns:
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def init_dist(launcher,
              backend='nccl',
              init_backend='torch',
              **kwargs) -> None:
    """Initialize distributed environment.

    Args:
        launcher (str): Way to launcher multi processes. Supported launchers
            are 'pytorch', 'mpi' and 'slurm'.
        backend (str): Communication Backends. Supported backends are 'nccl',
            'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    timeout = kwargs.get('timeout', None)
    if timeout is not None:
        # If a timeout (in seconds) is specified, it must be converted
        # to a timedelta object before forwarding the call to
        # the respective backend, because they expect a timedelta object.
        try:
            kwargs['timeout'] = datetime.timedelta(seconds=timeout)
        except TypeError as exception:
            raise TypeError(
                f'Timeout for distributed training must be provided as '
                f"timeout in seconds, but we've received the type "
                f'{type(timeout)}. Please specify the timeout like this: '
                f"dist_cfg=dict(backend='nccl', timeout=1800)") from exception
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, init_backend=init_backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, init_backend=init_backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, init_backend='torch', **kwargs) -> None:
    """Initialize distributed environment with PyTorch launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    rank = int(os.environ['RANK'])
    # LOCAL_RANK is set by `torch.distributed.launch` since PyTorch 1.1
    local_rank = int(os.environ['LOCAL_RANK'])
    if is_mlu_available():
        import torch_mlu  # noqa: F401
        torch.mlu.set_device(local_rank)
        torch_dist.init_process_group(
            backend='cncl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    elif is_npu_available():
        import torch_npu  # noqa: F401
        torch.npu.set_device(local_rank)
        torch_dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    elif is_musa_available():
        import torch_musa  # noqa: F401
        torch.musa.set_device(rank)
        torch_dist.init_process_group(
            backend='mccl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    else:
        torch.cuda.set_device(local_rank)

        if init_backend == 'torch':
            torch_dist.init_process_group(backend=backend, **kwargs)
        elif init_backend == 'deepspeed':
            import deepspeed
            deepspeed.init_distributed(dist_backend=backend, **kwargs)
        elif init_backend == 'colossalai':
            import colossalai
            colossalai.launch_from_torch(backend=backend, **kwargs)
        else:
            raise ValueError(
                'supported "init_backend" is "torch" or "deepspeed", '
                f'but got {init_backend}')


def _init_dist_mpi(backend, **kwargs) -> None:
    """Initialize distributed environment with MPI launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    if backend == 'smddp':
        try:
            import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'Please use an Amazon SageMaker DLC to access smdistributed: '
                'https://github.com/aws/deep-learning-containers/blob/master'
                '/available_images.md#sagemaker-framework-containers'
                '-sm-support-only') from e
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    torch_dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend,
                     port=None,
                     init_backend='torch',
                     **kwargs) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    # Not sure when this environment variable could be None, so use a fallback
    local_rank_env = os.environ.get('SLURM_LOCALID', None)
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
    else:
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(proc_id)

    if is_mlu_available():
        import torch_mlu  # noqa: F401
        torch.mlu.set_device(local_rank)
        torch_dist.init_process_group(backend='cncl', **kwargs)
    else:
        torch.cuda.set_device(local_rank)

        if init_backend == 'torch':
            torch_dist.init_process_group(backend=backend, **kwargs)
        elif init_backend == 'deepspeed':
            import deepspeed
            deepspeed.init_distributed(dist_backend=backend, **kwargs)
        elif init_backend == 'colossalai':
            import colossalai
            colossalai.launch_from_slurm(
                backend=backend,
                host=os.environ['MASTER_ADDR'],
                port=os.environ['MASTER_PORT'],
                **kwargs,
            )
        else:
            raise ValueError(
                'supported "init_backend" is "torch" or "deepspeed", '
                f'but got {init_backend}')
        

try:
    import torch_npu  # noqa: F401
    import torch_npu.npu.utils as npu_utils

    # Enable operator support for dynamic shape and
    # binary operator support on the NPU.
    npu_jit_compile = bool(os.getenv('NPUJITCompile', False))
    torch.npu.set_compile_mode(jit_compile=npu_jit_compile)
    IS_NPU_AVAILABLE = hasattr(torch, 'npu') and torch.npu.is_available()
except Exception:
    IS_NPU_AVAILABLE = False

    
def is_npu_available() -> bool:
    """Returns True if Ascend PyTorch and npu devices exist."""
    return IS_NPU_AVAILABLE


def is_mlu_available() -> bool:
    """Returns True if Cambricon PyTorch and mlu devices exist."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()


def is_mps_available() -> bool:
    """Return True if mps devices exist.

    It's specialized for mac m1 chips and require torch version 1.12 or higher.
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

try:
    import torch_musa  # noqa: F401
    IS_MUSA_AVAILABLE = True
except Exception:
    IS_MUSA_AVAILABLE = False

def is_musa_available() -> bool:
    return IS_MUSA_AVAILABLE