import torch
import numpy as np


# def copy_model_params_from_to(source, target):
#     for target_param, param in zip(target.parameters(), source.parameters()):
#         target_param.data.copy_(param.data)


# def cal_gradient_norm(parameters, norm_type=2):
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     parameters = list(filter(lambda p: p.grad is not None, parameters))
#     norm_type = float(norm_type)
#     if norm_type == torch._six.inf:
#         total_norm = max(p.grad.detach().abs().max() for p in parameters)
#     else:
#         total_norm = torch.norm(torch.stack(
#             [torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)

#     return total_norm


"""
GPU wrappers
"""

_use_gpu = False
device = None
_gpu_id = 0
_manual_seed = 1
dtype = torch.float32


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def set_random_seed(seed):
    global _manual_seed
    _manual_seed = seed
    torch.manual_seed(_manual_seed)
    np.random.seed(_manual_seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if gpu_enabled():
        torch.cuda.manual_seed(_manual_seed)
        torch.cuda.manual_seed_all(_manual_seed)


# noinspection PyPep8Naming


# def FloatTensor(*args, torch_device=None, **kwargs):
#     if torch_device is None:
#         torch_device = device
#     return torch.FloatTensor(*args, **kwargs).to(torch_device)


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs).to(device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs).to(device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs).to(device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)


def cov(tensor):
    return numpy2torch(np.cov(tensor.numpy()))


"""
data transform shape
"""


def torch2numpy(numpy_array):
    if numpy_array.is_cuda == True:
        numpy_array = numpy_array.cpu()
    return numpy_array.detach().numpy()


def list2numpy(list_data):
    return np.asarray(list_data)

# def from_numpy(*args, **kwargs):
#     return torch.from_numpy(*args, **kwargs).float().to(device)


def numpy2torch(torch_tensor):
    if device is not None:
        return torch.tensor(torch_tensor, dtype=dtype).cuda(device)
    else:
        return torch.tensor(torch_tensor, dtype=dtype)


def nn_1_to_n_n(nn_1):
    number_side = np.sqrt(nn_1.shape(0))
    n_n = nn_1.reshape(number_side, number_side)
    return n_n


def n_to_n_1(n):
    if torch.is_tensor(n):
        n_1 = n.reshape(n.shape[0], 1)
    return n_1
