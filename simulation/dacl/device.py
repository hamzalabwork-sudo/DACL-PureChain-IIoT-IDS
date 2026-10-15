"""Compute device for learning (cpu or cuda). Edge costs are always profiled on CPU (runner.edge_profile)."""
import torch

DEVICE = torch.device("cpu")


def set_device(name):
    global DEVICE
    DEVICE = torch.device(name)
    return DEVICE
