### This script trains the model
from lightning_net import load_lightning_data, LTGnet
import numpy as np

# Load the training data
X, y = load_lightning_data('lightning_test_data_100000.csv')

print(X.shape)
print(y.shape)

# Train the model
net = LTGnet(n_features = X.shape[1])
net.train(X, y, n_epochs=50, batch_size = 20)

# Save the model weights
net.save_model()
