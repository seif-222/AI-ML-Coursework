import sklearn
import pandas as pd
from sklearn.datasets import make_circles
n_samples = 1000
x,y=make_circles(n_samples,
                 noise=0.03,
                 random_state=43)


#DATA VISUALIZING
visual_panda=pd.DataFrame({"x1" : x[:,0], "x2" : x[:,1] , "y" : y})
print(visual_panda.head(20))


#BRAODER VISUAL
import matplotlib.pyplot as plt
plt.scatter(x=x[:,0],y=x[:,1],c=y ,
            cmap=plt.cm.RdYlBu);
plt.show()



#MAKE THE DATA AS SOME TENSORS AND MAKE THE SPLITS
import torch
x=torch.from_numpy(x).type(torch.float)
y=torch.from_numpy(y).type(torch.float)



# MAKE THE SPLITS
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)




#MAKING THE MODEL
from torch import nn
class model_of_nonlinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.l3(self.relu(self.l2(self.relu(self.l1(x)))))



#CALL THE MODEL AS NAME
model_1=model_of_nonlinear()



#MAKING THE OPTIMIZING FUN AND THE LOSS FN
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(),
                             lr=0.01)




#MAKING ACCURACY FUNCTION
def acc (y_pred,y_test):
    corr = torch.eq(y_pred,y_test).sum().item()
    acc =( corr / len(y_pred))*100
    return acc




#GETTING HTE STRONG VISUALIZING FUNCTION FRON THE LINK
import requests
from pathlib import Path
if Path("../helper_functions.py").is_file():
    print("helper_functions.py already exists")
else:
    print ("downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("../helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary



#MAKING THE TRAINING AND THE TESTING LOOP
epochs = 3000
for epoch in range(epochs):
    #MAKE TRAIN
    model_1.train()
    #FOREWARD PASS
    y_logits =model_1(x_train).squeeze()
    y_pred=torch.round(torch.sigmoid(y_logits))
    #LOSS AND ACC
    loss = loss_fn(y_logits,y_train) # """this takes the logits not the pred"""
    acc_train = acc(y_pred,y_train) #"""this takes the pred not logits"""
    #OPT ZERO
    optimizer.zero_grad()
    #
    loss.backward()
    #
    optimizer.step()



    #TESTING
    model_1.eval()
    with torch.inference_mode():
        yt_logits = model_1(x_test).squeeze()
        yt_pred = torch.round(torch.sigmoid(yt_logits))
        acc_test = acc(yt_pred,y_test)
        loss_test = loss_fn(yt_logits,y_test)

    if epoch % 100 == 0:
        print(f"ep: {epoch}, loss: {loss:.4f}, acc: {acc_train:.4f} , loss_test: {loss_test:.4f}, acc_test: {acc_test:.4f}")

        # PLOT DECISION BOUNDARIES
        if epoch % 500 == 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 2)
            plt.title("Test")
            plot_decision_boundary(model_1, x_test, y_test)  # model_3 = has non-linearity
            plt.show()
