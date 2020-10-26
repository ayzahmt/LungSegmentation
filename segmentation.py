import numpy as np


def create_mask(image):
    binary_image = np.array(image > -300, dtype=np.int8)
    return binary_image
