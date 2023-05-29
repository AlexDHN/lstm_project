#%%
import torch
import torch.nn as nn
from typing import Callable
from torch import Tensor, sigmoid, tanh, zeros, flatten, stack

class LSTMUnit(nn.Module):
    """ Class for a single LSTM unit

    Attributes:
        input_size (int): The dimension of the vector given as input
        output_size (int): The dimension of the vector returned in output
        input_number (int): The number of vectors given as input
        hidden_number (int): The number of hidden layer
        s (Tensor): The cell state ie the memory of the cell at time t
        s_candidate (Tensor): The candidate cell state
        h (Tensor): The output of the cell, also called hidden state
        weights_are_initialized (bool): If True initialize the state of the cell
        f (Tensor): The value for the forget gate
        i (Tensor): The value for the input gate
        o (Tensor): The value for the output gate
        Wf, Wi, Wo, Ws (Tensor, Tensor, Tensor, Tensor): The Weight matrices associeted with the input vector
        Uf, Ui, Uo, Us (Tensor, Tensor, Tensor, Tensor): The Weight matrices associeted with the output h
        bf, bi, bo, bs (Tensor, Tensor, Tensor, Tensor): The bias vectors

    """
    def __init__(self, input_size: int, output_size: int) -> None:
        """ Defines the parameters of an LSTM unit

        Args:
            input_size (int): The dimension of the vector given as input
            output_size (int): The dimension of the vector returned in output
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # The Weight matrices associeted with the input vector
        self.Wf = nn.Parameter(Tensor(input_size, output_size), requires_grad=True)
        self.Wi = nn.Parameter(Tensor(input_size, output_size), requires_grad=True)
        self.Wo = nn.Parameter(Tensor(input_size, output_size), requires_grad=True)
        self.Ws = nn.Parameter(Tensor(input_size, output_size), requires_grad=True)
        
        # The Weight matrices associeted with the output h
        self.Uf = nn.Parameter(Tensor(output_size, output_size), requires_grad=True)
        self.Ui = nn.Parameter(Tensor(output_size, output_size), requires_grad=True)
        self.Uo = nn.Parameter(Tensor(output_size, output_size), requires_grad=True)
        self.Us = nn.Parameter(Tensor(output_size, output_size), requires_grad=True)
        
        # The bias vectors
        self.bf = nn.Parameter(Tensor(output_size), requires_grad=True)
        self.bi = nn.Parameter(Tensor(output_size), requires_grad=True)
        self.bo = nn.Parameter(Tensor(output_size), requires_grad=True)
        self.bs = nn.Parameter(Tensor(output_size), requires_grad=True)
        
        self.weights_are_initialized = False  # The weights are not yet initialized
    
    def init_weights(self, f: Callable):
        """ Initialize the weights according to a distribution f defines

        Args:
            f (Callable): Function that associates to each weight a certain distribution
        """
        f(self)
        self.weights_are_initialized = True  # The weights are now initialized
        
    def forward(self, x: Tensor, init_memory: bool) -> Tensor:
        """ Calculates the output of the LSTM cell according to its state

        Args:
            x (Tensor): Input vector
            init_memory (bool): True if the memory must be initialized 

        Returns:
            (Tensor): Output of the cell
        """
        assert self.weights_are_initialized, 'The weights are not initialized'
        if init_memory:
            self.s, self.h = zeros(self.output_size), zeros(self.output_size)
        
        self.f = sigmoid(x @ self.Wf + self.h @ self.Uf + self.bf)
        self.i = sigmoid(x @ self.Wi + self.h @ self.Ui + self.bi)
        self.o = sigmoid(x @ self.Wo + self.h @ self.Uo + self.bo)
        self.s_candidate = tanh(x @ self.Ws + self.h @ self.Us + self.bs)
        self.s = self.f * self.s + self.i * self.s_candidate
        self.h = self.o * tanh(self.s)
        return self.h
        

class CustomLSTM(nn.Module):
    """ Allows to create the architecture of a neural network using layers of LSTM

    Attributes:
        input_size (int): The dimension of the vector given as input
        input_number (int): The number of vectors given as input
        hidden_number (int): The number of hidden layer
        hidden_size: (int): The dimension of the vector returned in LSTM output
        output_size (int): The dimension of the vector returned in output
        LSTMLayers (List): List of the layer
        dense (Callable): Put the output at the desire dimension
        
    """
    def __init__(self, 
                 input_size: int, # 29
                 input_number: int, # 3
                 hidden_number: int,  
                 hidden_size: int, # 100
                 output_size: int) -> None: # 29
        super().__init__()
        self.input_size = input_size
        self.input_number = input_number
        self.hidden_number = hidden_number
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.LSTMLayers = []
        self.LSTMLayers.append(LSTMUnit(input_size, hidden_size))  # Initialize the first layer with the dimension of the vector given as input
        for _ in range(hidden_number):
            self.LSTMLayers.append(LSTMUnit(hidden_size, hidden_size))  # Initialize the layer with dimension of the vector returned in LSTM output
        self.register_submodule_parameters()
        self.dense = nn.Linear(hidden_size * input_number, output_size)
        
        self.weights_are_initialized = False
    
    def register_submodule_parameters(self):
        """ Register the parameters of each layer in the neural network
        """
        for i, lstm in enumerate(self.LSTMLayers):
            self.register_parameter(f'_Wf_{i}', lstm.Wf)
            self.register_parameter(f'_Wo_{i}', lstm.Wo)
            self.register_parameter(f'_Ws_{i}', lstm.Ws)
            self.register_parameter(f'_Wi_{i}', lstm.Wi)
            self.register_parameter(f'_Uf_{i}', lstm.Uf)
            self.register_parameter(f'_Ui_{i}', lstm.Ui)
            self.register_parameter(f'_Uo_{i}', lstm.Uo)
            self.register_parameter(f'_Us_{i}', lstm.Us)
            self.register_parameter(f'_bf_{i}', lstm.bf)
            self.register_parameter(f'_bi_{i}', lstm.bi)
            self.register_parameter(f'_bo_{i}', lstm.bo)
            self.register_parameter(f'_bs_{i}', lstm.bs)
    
    def init_weights(self, f: Callable):
        """ Initialize the weights of each layer according to a distribution f defines

        Args:
            f (Callable): Function that associates to each weight a certain distribution
        """
        for lstm in self.LSTMLayers:
            lstm.init_weights(f)
        self.weights_are_initialized = True
        
    def forward(self, X:Tensor):
        """_summary_

        Args:
            X (Tensor): Take the features as input

        Returns:
            (Tensor): Output of the neural network
        """
        assert self.weights_are_initialized, 'The weights are not initialized'
        H = []
        for t in range(self.input_number):
            H.append(self.LSTMLayers[0].forward(X[:, t], init_memory=(t==0)))
        
        next_X = sigmoid(stack(H).swapaxes(0, 1))
        for hidden_layer in range(self.hidden_number):
            H = []
            for t in range(self.input_number):
                H.append(self.LSTMLayers[hidden_layer+1].forward(next_X[:, t], init_memory=(t==0)))
            next_X = sigmoid(stack(H).swapaxes(0, 1))
        return sigmoid(self.dense.forward(next_X.reshape(-1, self.hidden_size * self.input_number)))

    def save(self, path: str):
        """ Save the LSTM of the model
        """
        torch.save(self, path)  #torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str):
        """ Load the pretrain LSTM of the model
        """
        model = torch.load(path)  # .load_state_dict(torch.load(path))
        model.eval() 
        return model 
    
    
class ModifiedLSTM(nn.Module):
    """ Defining methods on PyTorch's own LSTM class to make it compatible with the rest of the code
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.input_size = kwargs["input_size"]
        self.hidden_size = kwargs["hidden_size"]
        self.input_number = kwargs["input_number"]
        self.hidden_number = kwargs["hidden_number"]
        self.output_size = kwargs["output_size"]
        
        self.weights_are_initialized = False
        self.LSTM = nn.LSTM(self.input_size, self.hidden_size, self.hidden_number)
        self.dense = nn.Linear(self.hidden_size * self.input_number, self.output_size)
        
    def forward(self, X: Tensor):
        return sigmoid(self.dense.forward(self.LSTM.forward(X)[0].reshape(-1, self.hidden_size * self.input_number)))
        
    def init_weights(self, f: Callable):
        """ Initialize the weights according to a distribution f defines

        Args:
            f (Callable): Function that associates to each weight a certain distribution
        """
        f(self)
        self.weights_are_initialized = True
    
    def save(self, path: str):
        """ Save the LSTM model

        Args:
            path (str): Save the model to the indicated path
        """
        torch.save(self, path)  #torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str):
        """ Load the pretrain LSTM of the model

        Args:
            path (str): Path where the model is located

        Returns:
            (CustomLSTM/ModifiedLSTM): Pretrain LSTM of the model
        """
        model = torch.load(path)  # .load_state_dict(torch.load(path))
        model.eval() 
        return model 