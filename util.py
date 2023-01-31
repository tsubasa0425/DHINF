import torch
import numpy as np

def one_hot(values, num_classes):
    batch_size = values.shape[0]
    seq_len = values.shape[1]
    result = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        for j in range(seq_len):
            if values[i][j] != -1:
                result[i][values[i][j]] = 1
            else:
                continue
    return result.long()

def masked_select(inputs, masks):
    result = []
    for i, mask in enumerate(masks, 0):
        if mask == 1:
            result.append(inputs[i])
    return np.array(result).astype(np.int32)