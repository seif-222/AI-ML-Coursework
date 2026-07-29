import random
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.nn.modules import loss
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm
import torchmetrics
import mlxtend


#GETTING THE DATA BASE
train_data=datasets.CIFAR10(root='./Data',train=True,transform=transforms.ToTensor(),download=True)
test_data=datasets.CIFAR10(root='./Data',train=False,transform=transforms.ToTensor(), download=True)
class_names = train_data.classes
# print(class_names)
class_names_ind = train_data.class_to_idx
# print(class_names_ind)
img,lab = train_data[0]
# print(img.shape)
# print(class_names[lab])



#VISUALIZE 1
plt.imshow(img.permute(1, 2, 0))
plt.title(lab)
plt.show()



#VISUALIZE 2
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
  random_idx = torch.randint(0, len(train_data), size=[1]).item()
  img, label = train_data[random_idx]
  fig.add_subplot(rows, cols, i)
  plt.imshow(img.permute(1,2,0), cmap="gray")
  plt.title(class_names[label])
  plt.axis(False);

plt.show()





# MAKING THE BATCHES USING THE DATA LOADER
batch_size= 32
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)
#SEE THE SHAPE
train_loader_image, train_loader_lab = next(iter(train_loader))
print(train_loader_image.shape)
print(train_loader_lab.shape)




#MAKE THE DEVICE AGNOSIC CODE
device = "cuda" if torch.cuda.is_available() else "cpu"



#MAKING A TIMER FUNCITION
def timer_fn(start,end,device):
    total_time = end - start
    print(f"total time on {device} : {total_time} seconds")
    return total_time




#MODEL 0
class linear_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers= nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )
    def forward(self,x):
        return self.layers(x)
model_0 = linear_model(3072 , 10 , len(class_names)).to(device)




#MODEL 1
class nonlinear_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers= nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.layers(x)

model_1 = nonlinear_model(3072 , 10 , len(class_names)).to(device)



#MODEL 2
class cnn_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.B1= nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=hidden_size,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.B2= nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_size*8*8, out_features=output_size),    #❎❎❎❎*8*8❎❎❎
        )

    def forward(self,x):
        x = self.B1(x)
        x = self.B2(x)
        return self.classifier(x)

model_2 = cnn_model(3 , 64 , len(class_names)).to(device)




#MAKE THE LOSS AND OPTIMIZER
loss_fn = nn.CrossEntropyLoss()


#GET THE ACCURACY FUNCTION
from helper_functions import accuracy_fn






#MAKING THE TRAINING FUNCTION
def training_fn(model,
                optimizer = torch.optim.Optimizer,
                train__loader=train_loader,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device = device):

    #INISIALIZE THE LOSS AND THE ACC
    loss_count , accuracy = 0, 0
    #PUT ON TRAIN
    model.train()
    #ITERATE AND MAKE DATA TO DEVICE
    for batch , (x,y) in enumerate(train__loader):
        x, y = x.to(device), y.to(device)
        #FORWARD
        y_logits = model(x)
        #LOSS & ACC
        loss=loss_fn(y_logits, y)
        loss_count += loss.item()
        accuracy += accuracy_fn(y, y_logits.argmax(dim=1))
        #
        optimizer.zero_grad()
        #
        loss.backward()
        #
        optimizer.step()
    #MAKE THE AVERAGE LOSS AND ACC THROUGH EACH BATCH
    l = len(train__loader)
    print(f"\ntrain_loss : {loss_count / l} ")
    print(f"train_accuracy : {accuracy / l} )")





#MAKE THE EVALUATION FUNCITON
def test_fn(model,
            test__loader=test_loader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device = device):
    model.eval()
    with torch.inference_mode():
        loss_count , accuracy = 0, 0
        for batch , (x,y) in enumerate(test__loader):
            x,y = x.to(device), y.to(device)
            y_logits = model(x)
            loss_count += loss_fn(y_logits, y)
            accuracy += accuracy_fn(y, y_logits.argmax(dim=1))
        l = len(test__loader)
        print(f"\ntest_loss : {loss_count / l} ")
        print(f"test_accuracy : {accuracy / l} )")



def eval_fn(model,
            test__loader=test_loader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device = device):
    model.eval()
    with torch.inference_mode():
        loss_count , accuracy = 0, 0
        for batch , (x,y) in enumerate(test__loader):
            x,y = x.to(device), y.to(device)
            y_logits = model(x)
            loss_count += loss_fn(y_logits, y)
            accuracy += accuracy_fn(y, y_logits.argmax(dim=1))
        l = len(test__loader)
        print(f"train_loss : {loss_count / l}")
        print(f"train_accuracy : {accuracy / l} )")
        #GET A MODEL DICTIONARY
        return {"model_name": model.__class__.__name__,
                "model_loss": (loss_count / l).item(),
                "model_acc": accuracy / l}




#
# from pathlib import Path
# # 1. SETUP THE PATH (At the top of your script)
# # Path() creates a "Path Object" which is smarter than a regular text string.
# # It automatically handles the difference between Windows (\) and Mac/Linux (/) slashes.
# MODEL_PATH = Path("../09_EFFICIENTNETB2_MODEL_TRAINED_10EPOCHES_20%PERCENT.pth")
#
# # mkdir = "make directory" (create folder).
# # parents=True: Creates any sub-folders needed to reach the path.
# # exist_ok=True: If the folder already exists, don't throw an error—just move on.
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
#
# # The / operator here isn't math; it's a Pathlib shortcut to join a folder name and a file name.
# MODEL_SAVE_PATH = MODEL_PATH / "CNN_model_data.pth"
#
# # 2. RELOAD (Place this before your training loop)
# # .exists() checks the hard drive to see if the file actually sits there.
# if MODEL_SAVE_PATH.exists():
#     # If the file is found, we begin the "Brain Transplant."
#     print(f"--- Loading saved model from: {MODEL_SAVE_PATH} ---")
#
#     # torch.load: Reads the binary Data from the file.
#     # map_location=device: Ensures the model loads correctly whether you are using a GPU (cuda) or CPU.
#     saved_state = torch.load(MODEL_SAVE_PATH, map_location=device)
#
#     # load_state_dict: Takes the dictionary of numbers and "pours" them into the model's layers.
#     model_2.load_state_dict(saved_state)
# else:
#     # If no file exists (first time running), the model stays with its random "starting" weights.
#     print("--- No saved model found, starting from scratch! ---")
#





# TRAINING MODEL 2
time_start = timer()

epochs = 20

optimizer = torch.optim.SGD(model_2.parameters(), lr=0.01)

for epoch in tqdm(range(epochs),desc="Epoch"):
    training_fn(model_2,optimizer)
    test_fn(model_2)

time_end = timer()
timer_fn(time_start, time_end,device)





#
#
# # 4. SAVE (Place this at the very end of your script)
# # This saves the "Memory" of the model after it has finished learning from the new epochs.
# print(f"--- Saving updated model to: {MODEL_SAVE_PATH} ---")
#
# # model_2.state_dict(): Extracts all the weights/parameters into a Python Dictionary.
# # torch.save: Freezes that dictionary into a file.
# # WARNING: This will overwrite the previous file so you always have the LATEST version.
# torch.save(obj=model_2.state_dict(), f=MODEL_SAVE_PATH)

#MAKING A PRED FUNCTION
def pred_fn(model,data,device="cpu"):#becasue NumPy does not support GPUs. Libraries like scikit-learn, matplotlib, and mlxtend (the stuff you use to build a Confusion Matrix) are built on top of NumPy so we need to make it on the cpu , doesnt have to be all of the thing just the final labels that we are going to use#becasue NumPy does not support GPUs. Libraries like scikit-learn, matplotlib, and mlxtend (the stuff you use to build a Confusion Matrix) are built on top of NumPy so we need to make it on the cpu , doesnt have to be all of the thing just the final labels that we are going to use

    pred = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for i in data :
            i=i.unsqueeze(dim=0).to(device)
            y_logits = model(i)
            pred.append(y_logits.argmax(dim=1))
        # torch.cat turns [tensor([3]), tensor([5])] -> tensor([3, 5])
        return torch.cat(pred)




#GET 9 RANDOM SAMPLES
# Pick 9 random indices first
random_indices = random.sample(range(len(test_data)), k=9)
# Pull only those 9 images
test_samples = [test_data[i][0] for i in random_indices]
test_samples_labels = [test_data[i][1] for i in random_indices]
# #SEE THE IMAGES
# plt.imshow(test_samples[0].permute(1, 2, 0))
# plt.title(f"Test Sample : {class_names[test_samples_labels[0]]}")
# plt.axis(False)



#MAKE A PREDICTION WITH THE MODEL
predicitons_of_model = pred_fn(model_2,test_samples)  #we made arg max in the main function (but it is better to have the propabilities and then at other tine use argmax and get the pred_labels


#VISUALIZING
plt.figure(figsize=(12, 12))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    # Matplotlib needs Permute.
    plt.imshow(sample.permute(1, 2, 0))
    # Convert numeric indices to actual text (e.g., 3 -> "Cat")
    pred_label = class_names[predicitons_of_model[i]]
    truth_label = class_names[test_samples_labels[i]]
    title_text = f"Pred: {pred_label}\nTruth: {truth_label}"
    # Visual feedback: Green for success, Red for failure
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")
    else:
        plt.title(title_text, fontsize=10, c="r")
    plt.axis(False)
plt.tight_layout() # remove the mess because plt can be stupid
plt.show()






#MAKING VISUALIZING FOR THE WHOLE MODEL PREFORMANCE USING THE CONFUSION MATRIX
y_preds = []
model_2.to(device)
model_2.eval()
with torch.inference_mode():
    for x,y in tqdm(test_loader,desc="Testing"):
        x,y = x.to(device), y.to(device)
        y_logits = model_2(x)
        y_pred=y_logits.argmax(dim=1)
        y_preds.append(y_pred.cpu())#becasue NumPy does not support GPUs. Libraries like scikit-learn, matplotlib, and mlxtend (the stuff you use to build a Confusion Matrix) are built on top of NumPy so we need to make it on the cpu
final_y_preds_tensor = torch.cat(y_preds)




# 2. Setup confusion instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names),task='multiclass')
confmat_tensor = confmat(preds=final_y_preds_tensor,
                         target=torch.tensor(test_data.targets))#turn it to tensor

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with numpy
    class_names=class_names,
    figsize=(10, 7)
)
plt.show()





