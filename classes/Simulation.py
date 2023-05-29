# %% Imports
import torch
import pandas as pd
import numpy as np

from time import time
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

from classes.LSTM import CustomLSTM, ModifiedLSTM
from classes.DataLoader import DataLoader

df = pd.read_csv("dataset.csv", index_col=0)
#df = (df - df.mean()) / df.std()
df.index = pd.to_datetime(df.index)
Y = np.array([(df.iloc[i] > df.iloc[i].median()).values for i in range(df.shape[0])])
#Y = np.array([(df.iloc[i] > df.median()).values for i in range(df.shape[0])])
df = df.drop(df.tail(1).index)

class Simulation:
    def __init__(self, **kwargs) -> None:
        self.learning_rate = kwargs["learning_rate"]
        self.batch_size = kwargs["batch_size"]
        self.num_epochs = kwargs["num_epochs"]
        self.window_size = kwargs["window_size"]
        self.hidden_number = kwargs["hidden_number"]
        self.hidden_size = kwargs["hidden_size"]
        self.weight_decay = kwargs["weight_decay"]
        self.override = kwargs["override"]
        
        self.df = df
        self.Y = Y
        
        if not self.override:
            if "path" in kwargs:
                self.LSTM = CustomLSTM.load(kwargs["path"])  # It is assumed that the characteristics of the loaded LSTM correspond to the input data
            else:
                self.LSTM = CustomLSTM(
                    input_size=df.shape[1], 
                    input_number=self.window_size, 
                    hidden_number=self.hidden_number, 
                    hidden_size=self.hidden_size, 
                    output_size=df.shape[1], 
                )
        else:
            if "path" in kwargs:
                self.LSTM = ModifiedLSTM.load(kwargs["path"])  # It is assumed that the characteristics of the loaded LSTM correspond to the input data
            else:
                self.LSTM = ModifiedLSTM(
                    input_size=df.shape[1],
                    input_number=self.window_size,
                    output_size=df.shape[1],
                    batch_first=True,
                    **kwargs, 
                )
        
        
        self.optimizer = Adam(
            params=self.LSTM.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay, 
        )
        self.criterion = torch.nn.MSELoss()
    
    def make_dataloaders(self, test_train_split: float = 0.8):
        pivot_index = round(df.shape[0] * test_train_split)
        
        self.train_dataloader = lambda : DataLoader(
            #df=df.iloc[:pivot_index], 
            df=(df.iloc[:pivot_index] - df.iloc[:pivot_index].mean()) / df.iloc[:pivot_index].std(),
            Y=Y[:pivot_index],
            window_size=self.window_size, 
            batch_size=self.batch_size, 
        )
        
        self.test_dataloader = lambda : DataLoader(
            #df=df.iloc[pivot_index:],
            df=(df.iloc[pivot_index:] - df.iloc[pivot_index:].mean()) / df.iloc[pivot_index:].std(), 
            Y=Y[pivot_index:],
            window_size=self.window_size, 
            batch_size=self.batch_size, 
        )
        
        self.dataloaders = {
            "train": self.train_dataloader, 
            "test": self.test_dataloader, 
        
        }

        return pivot_index
    
    def train(self, verbose: int = 1):
        """ Main training loop
        
        Args:
            verbose (int, optional): 0: no output, 1: epochs tqdm, 2: prints losses per epoch. Defaults to 1.
        """
        self.train_loss_history = []
        self.test_loss_history = []
        main_loop = tqdm(range(self.num_epochs)) if verbose == 1 else range(self.num_epochs)
        for epoch in main_loop:
            start_time = time()
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.LSTM.train()
                else:
                    self.LSTM.eval()
                loss_sum = 0 # Total loss
                loss_num = 0 # To compute average loss
                
                if phase == 'train':
                    iterator = tqdm(self.dataloaders[phase]()) if verbose == 2 else self.dataloaders[phase]()
                else:
                    iterator = self.dataloaders[phase]()
                for batch in iterator:
                    x, y_true = batch
                    if phase == 'train' and verbose == 2:
                        iterator.set_description(f"Epoch {epoch+1}/{self.num_epochs}")
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase=='train'):
                        
                        y_pred = self.LSTM(x) if not self.override else self.LSTM(x)[0]
                        loss = self.criterion(y_pred, y_true)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        
                        loss_sum += loss.item()
                        loss_num += 1
                
                # Print loss if verbose == 2
                if verbose == 2:
                    if phase == 'train':
                        time_elapsed = time() - start_time
                        print(f'Epoch {epoch+1}/{self.num_epochs} complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
                    print("  ==> {} loss: {:.4f}".format(phase.capitalize(), loss_sum / loss_num * 100))
                    print()
                
                # Save loss history
                if phase == 'train':
                    self.train_loss_history.append(loss_sum / loss_num)
                else:
                    self.test_loss_history.append(loss_sum / loss_num)
            if verbose==1:
                main_loop.set_description(f"Train loss: {self.train_loss_history[-1]*100:.4f} | Test loss: {self.test_loss_history[-1]*100:.4f}")
            

    def confusion_matrix(self, display: bool = True):
        """ Returns the confusion matrix for the training and test sets
        Output is either a Numpy matrix; or a plot object if display is 'True'
        """
        
        train_confusion_matrix = np.zeros((2, 2))
        test_confusion_matrix = np.zeros((2, 2))
        for phase in ['train', 'test']:
            iterator = tqdm(self.dataloaders[phase]()) if phase == 'train' else self.dataloaders[phase]()
            for batch in iterator:
                x, y_true = batch
                with torch.no_grad():
                    
                    y_pred = (self.LSTM(x) > 0.5).int()
                    if phase == 'train':
                        train_confusion_matrix += confusion_matrix(y_true.flatten(), y_pred.flatten())
                    else:
                        test_confusion_matrix += confusion_matrix(y_true.flatten(), y_pred.flatten())
        
        self.train_confusion_matrix = train_confusion_matrix
        self.test_confusion_matrix = test_confusion_matrix
        
        if display:
            return (
                ConfusionMatrixDisplay(train_confusion_matrix), 
                ConfusionMatrixDisplay(test_confusion_matrix)
            )
        return train_confusion_matrix, test_confusion_matrix

    def classification_report(self):
        """ Returns the classification report for the training and test sets
        """
        
        self.train_accuracy = (self.train_confusion_matrix[0, 0] + self.train_confusion_matrix[1, 1]) / self.train_confusion_matrix.sum()
        self.test_accuracy = (self.test_confusion_matrix[0, 0] + self.test_confusion_matrix[1, 1]) / self.test_confusion_matrix.sum()
        
        self.train_recall = self.train_confusion_matrix[1, 1] / (self.train_confusion_matrix[1, 1] + self.train_confusion_matrix[1, 0])
        self.test_recall = self.test_confusion_matrix[1, 1] / (self.test_confusion_matrix[1, 1] + self.test_confusion_matrix[1, 0])
        
        self.train_precision = self.train_confusion_matrix[1, 1] / (self.train_confusion_matrix[1, 1] + self.train_confusion_matrix[0, 1])
        self.test_precision = self.test_confusion_matrix[1, 1] / (self.test_confusion_matrix[1, 1] + self.test_confusion_matrix[0, 1])
        
        self.train_f1_score = 2 * self.train_precision * self.train_recall / (self.train_precision + self.train_recall)
        self.test_f1_Score = 2 * self.test_precision * self.test_recall / (self.test_precision + self.test_recall)
        
        print("{:<8} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format("Phase", "Accuracy", "Recall", "Precision", "F1 Score", "TP", "TN"))
        print("{:<8} {:<15.2%} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.0f} {:<10.0f}".format("Train", self.train_accuracy, self.train_recall, self.train_precision, self.train_f1_score, self.train_confusion_matrix[1, 1], self.train_confusion_matrix[0, 0]))
        print("{:<8} {:<15.2%} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.0f} {:<10.0f}".format("Test", self.test_accuracy, self.test_recall, self.test_precision, self.test_f1_Score, self.test_confusion_matrix[1, 1], self.test_confusion_matrix[0, 0]))

    def plot_loss(self):
        """ Plots the loss history for the training and test sets
        """
        plt.plot(self.train_loss_history, label="Train")
        plt.plot(self.test_loss_history, label="Test")
        plt.legend()
        plt.show()

# %%
