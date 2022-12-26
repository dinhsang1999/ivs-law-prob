import numpy as np
import random
import sys
import torch

def set_seed(seed=0):
    '''
    Seed for randomize
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True