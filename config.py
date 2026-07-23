import platform
import numpy as np
import torch

print(platform.platform())
print(platform.processor())

print(torch.__version__)
print(torch.__config__.show())

print(np.__version__)
np.show_config()
