# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097

import torch, numpy, random

def ensure_reproducibility(seed, debug_only=True):

    if not seed:
        seed = 10

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    if debug_only:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # apparently slows down trading, so only use if debugging
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
     


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

