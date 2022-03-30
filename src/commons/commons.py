import torch
 
def move_batch(batch, device):
    for k, v in batch.items():
        if type(v) is not torch.Tensor:
            continue
        v = v.to(device)
        batch[k] = v
    return batch