import torch
import pandas as pd
import numpy as np


class DataLoader:
    """DataLoader iterator class to provide model with batches of data
    """

    def __init__(
        self, df: pd.DataFrame, Y: list, window_size: int, batch_size: int
    ) -> None:
        self.df = df
        self.Y = Y
        self.batch_size = batch_size
        self.window_size = window_size
        self.counter = 0
        self.n = self.df.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.n:
            self.counter += self.batch_size
            batch_X = np.array(
                [
                    self.df.iloc[i : i + self.window_size].values
                    for i in range(self.counter - self.batch_size, self.counter)
                    if i + self.window_size + 1 <= self.n
                ]
            )

            batch_Y = np.array(
                [
                    self.Y[i + self.window_size]
                    for i in range(self.counter - self.batch_size, self.counter)
                    if i + self.window_size + 1 <= self.n
                ]
            )
            return torch.Tensor(batch_X), torch.Tensor(batch_Y)

        raise StopIteration

    def __len__(self):
        return self.n // self.batch_size

        """
        def __next__(self):
        if self.counter < self.n:
            self.counter += self.batch_size
            return [(self.df.iloc[i : i+self.window_size], self.Y[i+self.window_size]) \
                for i in range(self.counter-self.batch_size, self.counter) \
                    if i + self.window_size + 1 <= self.n]
        raise StopIteration
        """
