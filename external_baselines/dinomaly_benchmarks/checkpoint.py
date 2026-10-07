"""Rolling compact checkpoint plus genuine shuffled-iterator and RNG recovery."""
import os
import random
from pathlib import Path


def rng(device):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None)


def restore_rng(state, device):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def cursor_valid(cursor, length):
    if length <= 0 or cursor['epoch'] < 0 or not 0 <= cursor['batches'] <= length:
        raise ValueError('Invalid saved loader cursor')


class Stream:
    """Replay deterministic RGB transforms and official shuffled epoch order.

    Save RNG just before iterator creation. On resume, recreate the epoch and
    discard consumed batches, then restore post-update dropout RNG. Workers have
    deterministic transforms; no stochastic augmentation/persistent workers.
    """
    def __init__(self, loader, device, cursor=None, current_rng=None):
        self.loader, self.device = loader, device
        if cursor is None:
            self.state = dict(epoch=0, batches=0, start_rng=rng(device))
            self.iterator = iter(loader)
        else:
            cursor_valid(cursor, len(loader))
            self.state = cursor.copy()
            restore_rng(cursor['start_rng'], device)
            self.iterator = iter(loader)
            for _ in range(cursor['batches']):
                next(self.iterator)
            restore_rng(current_rng, device)

    def next(self):
        try:
            value = next(self.iterator)
        except StopIteration:
            self.state = dict(epoch=self.state['epoch']+1, batches=0, start_rng=rng(self.device))
            self.iterator = iter(self.loader)
            value = next(self.iterator)
        self.state['batches'] += 1
        return value


def compact(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')}


def reload(model, weights):
    current = model.state_dict()
    expected = {k for k in current if not k.startswith('encoder.')}
    if set(weights) != expected:
        raise ValueError('Missing/unexpected compact model keys')
    model.load_state_dict({k:current[k] if k.startswith('encoder.') else weights[k] for k in current}, strict=True)


def save(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)
