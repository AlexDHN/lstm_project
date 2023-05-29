from classes.Simulation import Simulation
from utils.init_functions import *
import pandas as pd
import torch


def strategy(data: pd.DataFrame, init_functions: list, **kwargs):
    """ Computes and returns the trading strategy described by the paper

    Args:
        data (DataFrame): Dataset of yield logs
        init_functions (list): The list of functions to be considered for the strategy
        kwargs (dictionary): The parameters for the implementation/loading of the neural network

    Returns:
        (tuple(Dataframe)): Tuple containing raw data and trading strategy
    """
    trading = {}

    for func in init_functions:
        if "path" in kwargs:
            kwargs["path"] = "models/{}".format(func)
            trading[func] = Simulation(**kwargs)
            pivot_index = trading[func].make_dataloaders()
        else:
            trading[func] = Simulation(**kwargs)
            trading[func].make_dataloaders()
            trading[func].LSTM.init_weights(eval(func))
            trading[func].train(verbose=1)
            trading[func].LSTM.save("models/{}".format(func))
            print("Simulation {} trained.".format(func))

    Y_pred = {}
    for func in trading:
        trade = trading[func]
        Y_pred_aux = []
        iterator = trade.dataloaders["test"]()
        for batch in iterator:
            x, y_true = batch
            y_pred = trade.LSTM(x) if not trade.override else trade.LSTM(x)[0]
            Y_pred_aux.append(y_pred)

        Y_pred_aux = torch.cat(Y_pred_aux, 0)
        Y_pred_aux = (Y_pred_aux > 0.5).int()
        Y_pred[func] = Y_pred_aux

    seuil = len(init_functions) // 2
    y_pred = sum(Y_pred.values())
    y_pred = (y_pred > seuil).int()
    data = data.iloc[pivot_index + kwargs["window_size"] + 1:]
    df_pred = pd.DataFrame(y_pred.numpy(), index=data.index, columns=data.columns)

    return data, df_pred
