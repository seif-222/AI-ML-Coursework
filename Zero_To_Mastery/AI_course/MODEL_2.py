import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from helper_functions import plot_predictions, plot_decision_boundary

#MAKE THE DATA
NO_CLASSES = 4
NO_FEATURES = 2
RANDOM_SEED = 43

x,y = make_blobs(n_samples=1000,
                 n_features=NO_FEATURES,
                 centers=NO_CLASSES,
                 cluster_std=1.5,
                 random_state=RANDOM_SEED)

#TURN TO TENSORS
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.long)

#MAKE THE SPLIT
x_train , x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = RANDOM_SEED)


#MAKE THE VISUAL DATA
plt.figure(figsize=(10, 7))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu);
plt.show()


#MAKE THE MODEL
class blobmodel(nn.Module):
    def __init__(self, input_features, output_features,hidden_layers):
        super().__init__()
        self.layes = nn.Sequential(
            nn.Linear(input_features, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, output_features),
        )
    def forward(self, x):
        return self.layes(x)



#MAKE THE NAME FOR THE MODEL
model_2 = blobmodel(NO_FEATURES,NO_CLASSES,256)


#MAKE THE LOSS FUNCTION AND THE OPTIMIZER
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_2.parameters(), lr = 0.01)



#MAKING ACCURACY FUNCTION
def acc (y_pred,y_test):
    corr = torch.eq(y_pred,y_test).sum().item()
    acc =( corr / len(y_pred))*100
    return acc



#MAKE THE TRAINING LOOP FOR MODEL
epochs = 2000
for epoch in range(epochs):
    model_2.train()
    #MAKE FOREWARD
    y_logits = model_2(x_train)
    y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
    #MAKE LOSS,ACC
    loss = loss_function(y_logits, y_train)
    acc_train = acc(y_preds,y_train)
    #ZERO GRAD
    optimizer.zero_grad()
    #BACK
    loss.backward()
    #
    optimizer.step()


    #MAKE THE TESTS
    model_2.eval()
    with torch.inference_mode():
        y_test_logits = model_2(x_test)
        y_test_preds = torch.argmax(torch.softmax(y_test_logits, dim=1), dim=1)
        loss_test = loss_function(y_test_logits, y_test)
        acc_test = acc(y_test_preds,y_test)

    if epoch % 100 == 0:
        print(f"epoches: {epoch:.3f}, loss_train: {loss:.3f}, acc_train: {acc_train:.3f} , loss_test: {loss_test:.3f}, acc_test: {acc_test:.3f}")
        if epoch % 500 == 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 2)
            plt.title("Test")
            plot_decision_boundary(model_2, x_test, y_test)  # model_3 = has non-linearity
            plt.show()


