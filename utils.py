"""Shared runtime and model-loading helpers."""

import os
import random
import re

from nnsight import LanguageModel
import numpy as np
import torch
from transformers import set_seed as transformers_set_seed


def sanitize_model_id(model_name_or_path):
    """Convert a model name or path into an artifact-safe identifier."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name_or_path).strip("/"))


def set_seed(seed):
    """Seed supported random number generators and deterministic CUDA settings."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transformers_set_seed(seed)


def load_nnsight_model(
    model_name,
    revision,
    device,
    seed,
    dtype=None,
):
    """Load one model revision through NNsight."""
    set_seed(seed)
    print(f"Loading Model Name: {model_name}, Revision: {revision}")
    model = LanguageModel(
        model_name,
        revision=revision,
        device_map=device,
        tokenizer=None,
        dispatch=True,
    )
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    return model
