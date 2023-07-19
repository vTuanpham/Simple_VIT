import sys
import os
import gc
import time
import random
import argparse
sys.path.insert(0,r'./')
from functools import wraps

import torch
import numpy as np
import matplotlib.pyplot as plt


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')

        return result
    return timeit_wrapper


def set_seed(value):
    print("\n Random Seed: ", value)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(value)


def clear_cuda_cache():
    print("\n Clearing cuda cache...")
    torch.cuda.empty_cache()
    print("\n Running garbage collection...")
    gc.collect()


def plot_image(tensor, label: str=None):
    # Convert the tensor to a numpy array
    images = tensor.numpy()

    # Normalize the pixel values to [0, 1]
    images = np.clip(images, 0, 1)

    # Reshape the tensor to (batch, height, width, channels)
    images = np.transpose(images, (0, 3, 2, 1))

    # Iterate over each image in the batch
    for idx, image in enumerate(images):
        # Remove the batch dimension if it exists
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)

        # Convert image to RGB if channel order is BGR
        if image.shape[-1] == 3 and np.max(image) > 1:
            image = image[..., ::-1]  # Reverse channel order

        # Plot the image
        plt.imshow(np.rot90(image, 3))
        if label is not None: plt.title(label[idx])
        plt.axis('off')
        plt.show()


class TwoWayDict:
    def __init__(self, dict):
        self.dict = dict
        self.dict.update({item[1]: item[0] for item in  self.dict.items()})
