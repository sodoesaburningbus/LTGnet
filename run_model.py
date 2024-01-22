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

# Mask to only confident areas
y_pred = np.squeeze(y_pred.detach().numpy())
mask = (y_pred > 0.9) | (y_pred < 0.2)
y_pred = y_pred[mask]
y_test = y_test[mask]

fn = open('nifty_rows.txt', 'w')
rows = np.arange(100000)[mask]
for r in rows:
    fn.write(f'{r}\n')
fn.close()

# Print performance
y_pred = np.array(np.round(y_pred), dtype='bool')
y_test = np.array(np.squeeze(y_test), dtype='bool')

tp = np.sum(y_pred & y_test)
tn = np.sum((~y_pred) & (~y_test))
fp = np.sum(y_pred & (~y_test))
fn = np.sum((~y_pred) & y_test)

# Print to log
print(f'\nTrue Positives = {tp}')
print(f'\nTrue Negatives = {tn}')
print(f'\nFalse Positives = {fp}')
print(f'\nFalse Negatives = {fn}')
print(f'\nAccuracy = {(tp+tn)/(tp+tn+fp+fn)*100.0:.2f}%')
print(f'\nPoD = {tp/(tp+fn)*100.0:.2f}%')
print(f'\nFAR = {fp/(fp+tn)*100.0:.2f}%')