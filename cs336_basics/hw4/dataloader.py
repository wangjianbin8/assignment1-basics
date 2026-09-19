import random
import numpy as np
import torch

class DataLoader:
    def __init__(self, data, batch_size,
            context_length, shuffle=True
        ):
        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_length = context_length

    def get_train_batch_data(self):
        #先选择初始位置，共batch_size个
        idxs = np.random.randint(0, self.data_len - self.context_length, size=(self.batch_size))
        x = np.stack([self.data[i: i + self.context_length] for i in idxs])
        y = np.stack([self.data[i + 1: i + self.context_length + 1] for i in idxs])
        return torch.tensor(x), torch.tensor(y)
        '''
        x.shape = [batch_size, context_length]
        y.shape = [batch_size, context_length]  
        '''

    def get_valid_batch_data_iter(self):
        max_start = self.data_len - self.context_length - 1
        starts = np.arange(max_start + 1) #所有合法的起始位置

        for i in range(0, len(starts), self.batch_size):
            batch_idxs = starts[i: i + self.batch_size]
            #当前batch的所有起点，一共有self.batch_size个

            if len(batch_idxs) < self.batch_size:
                break
            x = np.stack([
                self.data[j: j + self.context_length]
                for j in batch_idxs
            ])

            y = np.stack([
                self.data[j + 1: j + self.context_length + 1]
                for j in batch_idxs
            ])

            yield (
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            )

    def __len__(self):
        num_samples = self.data_len - self.context_length - 1
        return num_samples // self.batch_size