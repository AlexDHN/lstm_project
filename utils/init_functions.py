import torch.nn as nn


def random_normal(lstm_cell, mu=0, sigma=0.1):
    for weight in lstm_cell.parameters():
        nn.init.normal_(weight, mean=mu, std=sigma)


def random_uniform(lstm_cell, a=-0.05, b=0.05):
    for weight in lstm_cell.parameters():
        nn.init.uniform_(weight, a=a, b=b)


def truncated_normal(lstm_cell, mu=0, sigma=0.05, a=-2.0, b=2.0):
    for weight in lstm_cell.parameters():
        nn.init.trunc_normal_(weight, mean=mu, std=sigma, a=a, b=b)


def zeros(lstm_cell):
    for weight in lstm_cell.parameters():
        nn.init.constant_(weight, 0)


def ones(lstm_cell):
    for weight in lstm_cell.parameters():
        nn.init.constant_(weight, 1)


def glorot_normal(lstm_cell, gain=1.0):
    for weight in lstm_cell.parameters():
        if weight.ndim == 2:
            nn.init.xavier_normal_(weight, gain=gain)
        elif weight.ndim == 1:
            nn.init.xavier_normal_(weight.reshape(len(weight), 1), gain=gain)


def glorot_uniform(lstm_cell, gain=1.0):
    for weight in lstm_cell.parameters():
        if weight.ndim == 2:
            nn.init.xavier_uniform_(weight, gain=gain)
        elif weight.ndim == 1:
            nn.init.xavier_uniform_(weight.reshape(len(weight), 1), gain=gain)


def identity(lstm_cell):
    for weight in lstm_cell.parameters():
        if weight.ndim == 2:
            nn.init.eye_(weight)
        elif weight.ndim == 1:
            nn.init.eye_(weight.reshape(len(weight), 1))


def orthogonal(lstm_cell):
    for weight in lstm_cell.parameters():
        if weight.ndim == 2:
            nn.init.orthogonal_(weight)
        elif weight.ndim == 1:
            nn.init.orthogonal_(weight.reshape(len(weight), 1))


def constant_(lstm_cell, c=0.05):
    for weight in lstm_cell.parameters():
        nn.init.constant_(weight, c)


def variance_scaling(lstm_cell, mode="fan_in"):
    for weight in lstm_cell.parameters():
        if weight.ndim == 2:
            nn.init.kaiming_normal_(weight, mode=mode)
        elif weight.ndim == 1:
            nn.init.kaiming_normal_(weight.reshape(len(weight), 1), mode=mode)
