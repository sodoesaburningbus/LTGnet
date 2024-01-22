### This code runs the inference over the AI model
### Christopher Phillips

# Import modules
from lightning_net import load_lightning_data, LTGnet
import numpy as np

# Load the evaluation data
X, y_test = load_lightning_data('lightning_test_data_100000.csv')

# Load the model
net = LTGnet(n_features = X.shape[1])
net.load_model('classifier_2024-01-20_2127.pt')

# Run predictions
y_pred = net.predict(X)

# Use simple rounding to convert probabilistic prediction to True/False
y_pred = np.array(np.round(np.squeeze(y_pred.detach().numpy())), dtype='bool')
