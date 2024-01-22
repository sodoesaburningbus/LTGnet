### This class is a NueralNet for lightning classification
### It also contains the data loader
### Christopher Phillips

### Import required modules
import copy
from datetime import datetime
import pandas
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

### The data loader
def load_lightning_data(fpath):

    df = pandas.read_csv(fpath, header=None)
    X = np.array(df.iloc[:,2:-1])
    y = np.array(df.iloc[:,0])

    X = normalize(X, axis=0)
    return X, y


### The Neural Network
class LTGnet(nn.Module):

    # The initialization function
    def __init__(self, n_features):

        # Need to call the init of the base Module function
        super(LTGnet, self).__init__()

        # Define the model architecture
        # This is more art than science, unfortunately.
        # Multiple hidden layers are called Deep Learning
        # More layers typically work better than fewer but larger layers.
        self.model = nn.Sequential(
            nn.Linear(n_features, 60), # This is the input layer of the model
            nn.LeakyReLU(),            # This is the activation function, it introduces non-linearity
            nn.Linear(60, 30),         # This is the first hidden layer of the model
            nn.LeakyReLU(),            # Activation function for first hidden layer
            nn.Linear(30, 20),         # Second hidden layer
            nn.LeakyReLU(),            # Activation function for second hidden layer
            nn.Linear(20, 1),          # The output layer of the model
            nn.Sigmoid()               # Another activation function, mapping the output to [0, 1]
        )

        # Set the optimizer
        # The optimizer is what adjusts the model weights on each pass
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Set the Loss function
        # The Loss function is the function that is MINIMIZED during training.
        # RMSE, MBE, and MAE are all used, as are more esoteric functions that
        # are custom built. This one is binary cross entropy which is often used
        # for classification tasks when the model outputs a probability.
        self.loss_fn = nn.BCELoss()

    ### Define the training function
    ### Training typically happens by breaking the input data into batches
    ### and training on each batch in sequence. This is repeated for a defined number of
    ### Epochs.
    ### Inputs:
    ###   X, the inputs as a numpy array (samples, features)
    ###   y, the training labels as a numpy array (samples,)
    ###   n_epochs, the number of training epochs
    ###   batch_size, the size of each batch
    def train(self, X, y, n_epochs=100, batch_size=50, log_path=None):

        # Create the training/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True)

        # Convert to PyTorch Tensors (in case they're not already)
        # Note the 32bit dtype. Model weights default to float32 and the data must be the same
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Get training start date
        start_date = datetime.utcnow()

        # Create the log file
        if (log_path == None):
            log_path = f'training_log_{start_date.strftime("%Y-%m-%d_%H%M")}.txt'
        log = open(log_path, 'w')
        log.write(f'Number of epochs: {n_epochs}\nBatch size: {batch_size}\nStart Date: {start_date.strftime("%Y-%m-%d_%H%M")}\nEpoch,, Loss')

        # Perform the actual training
        best_loss = np.inf # Initialize a best loss metric
        for epoch in range(n_epochs): # The Epoch loop. Each Epoch will loop through all training data
            
            # Set the model to training mode (some layers behave differntly during training)
            self.model.train()

            # The batch loop, break the data into chunks for training
            for start in torch.arange(0, len(X_train), batch_size):

                # Get training batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]

                # Do a forward pass through the model
                # This is where the model makes prediction
                y_pred = self.model(X_batch)

                # Compute the loss
                loss = self.loss_fn(y_pred, y_batch.unsqueeze(1))

                # Back propgation
                # This is where the model evaluates it performance
                self.optimizer.zero_grad()
                loss.backward()

                # Update the model weights
                self.optimizer.step()

            # Evaluate model accuracy after the Epoch
            self.model.eval() # Set model to evaluation mode
            y_pred = self.model(X_test)
            loss = self.loss_fn(y_pred, y_test.unsqueeze(1))
            loss = float(loss)
            log.write(f'\n{epoch},{loss:.4f}')

            # Check if best accuracy
            if (loss < best_loss):
                best_loss = loss
                best_weights = copy.deepcopy(self.model.state_dict())

        # Get the training end date
        end_date = datetime.utcnow()

        # After training, resotre model to best performance
        self.model.load_state_dict(best_weights)

        # Get error statistics (true positive, etc.)
        y_pred = np.array(np.round(np.squeeze(self.model(X_test).detach().numpy())), dtype='bool')
        y_test = np.array(np.round(np.squeeze(y_test.detach().numpy())), dtype='bool')

        tp = np.sum(y_pred & y_test)
        tn = np.sum((~y_pred) & (~y_test))
        fp = np.sum(y_pred & (~y_test))
        fn = np.sum((~y_pred) & y_test)

        # Print to log
        log.write(f'\nTraining ended at {end_date.strftime("%Y-%m-%d_%H%M")}')
        log.write(f'\nTotal training time: {(end_date-start_date).total_seconds()} s')
        log.write(f'\nBest training loss: {best_loss:.4f}')
        log.write(f'\nTrue Positives = {tp}')
        log.write(f'\nTrue Negatives = {tn}')
        log.write(f'\nFalse Positives = {fp}')
        log.write(f'\nFalse Negatives = {fn}')
        log.write(f'\nAccuracy = {(tp+tn)/(tp+tn+fp+fn)*100.0:.2f}%')
        log.write(f'\nPoD = {tp/(tp+fn)*100.0:.2f}%')
        log.write(f'\nFAR = {fp/(fp+tn)*100.0:.2f}%')
        log.close()

    # A function to load a previous model state
    def load_model(self, fpath):
        self.model.load_state_dict(torch.load(fpath))
        self.model.eval()

    # A function to save the current model state
    def save_model(self, spath=None):
        if (spath == None):
            spath = f"checkpoint_{datetime.utcnow().strftime('%Y-%m-%d_%H%M')}.pt"

        torch.save(self.model.state_dict(), spath)

    # A function for inferrencing
    def predict(self, X):

        X = torch.tensor(X, dtype=torch.float32)
        return self.model(X)