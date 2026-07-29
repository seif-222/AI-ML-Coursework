import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np

#DATA:

weight = 4
bias = 2
x = torch.arange(0,1,0.01).unsqueeze(1)
y = x * weight + bias

# DATA CLASS:

n = int(0.8*len(x))
tx , ty = x[:n] , y[:n]
tsx , tsy = x[n:] , y[n:]


#VISUALS:

def visual_data (tx , ty , tsx , tsy , pre = None):
    plt.figure(figsize=(10,7))
    plt.scatter(tx , ty , c= "b" , s=4 , label ="training Data")
    plt.scatter(tsx , tsy , c= "r" , s=4 , label ="testing Data")
    if pre is not None:
        plt.scatter(tsx , pre , c= "g" , s=4 , label ="predicted Data")

    plt.legend(prop={'size': 10})
    plt.show()




#BUILDING THE MODEL:

class modeloflinear (nn.Module):
    def __init__(self):
        super().__init__()

        """
        self.weights = nn.Parameter(torch.randn(1
                                                ,requires_grad=True
                                                ,dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1
                                             ,requires_grad=True
                                             ,dtype=torch.float32))
                                             """

        #INSTEAD OF MAKING THIS STUFF MANUAL WE USE:
        self.linear_layer = nn.Linear(in_features= 1,
                                      out_features= 1)

        """
    def forward(self, x:torch.Tensor):
        return  self.weights*x + self.bias
        """

    #INSTEAD OF MAKING THIS STUFF MANUAL WE USE:
    def forward (self,x:torch.Tensor) -> torch.Tensor:
        return  self.linear_layer(x)


#SETUP:

torch.manual_seed(34)
model = modeloflinear()
# print(model.state_dict())



#TRAINING SETUP:

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)


#LISTS:

i_count =[]
trainloss=[]
testloss=[]



#TRAINING LOOP:

torch.manual_seed(34)

for i in range(251):
    #train mode
    model.train()
    #foreward pass
    pre = model(tx)
    #loss
    loss = loss_fn(pre , ty)
    #opt . zero grad
    optimizer.zero_grad()
    #backpropagation
    loss.backward()
    #gradient decent
    optimizer.step()
    # this is the testing that we are making inside the loop to see what is happeneing
    if i % 10 == 0:#to see every 10 steps
        # TEST stop the training to have model tested
        model.eval()
        with torch.inference_mode():
            pre = model(tsx)
            tloss = loss_fn(pre , tsy)

        #FOR THE ENDING GRAPH
        i_count.append(i)
        trainloss.append(loss.detach().numpy())
        testloss.append(tloss.detach().numpy())

        # FOR DIRECT TRACKING AND VISUALIUSING
        print(f"loss: {loss} | i = {i} ")
        if i %50 == 0:
            visual_data(tx , ty , tsx , tsy , pre )
            print(model.state_dict())



# PLOT THE LOSE CURVES:

plt.plot(i_count, trainloss, label="Train loss")
plt.plot(i_count, testloss, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

#
# # SAVING THE MODEL:
#
# from pathlib import Path
# #CREATE MODEL DICTIONARY
# MODEL_PATH=Path("09_EFFICIENTNETB2_MODEL_TRAINED_10EPOCHES_20%PERCENT.pth")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
# #CREATE SAVE PATH
# MODEL_NAME = "model_0 Data.pth"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# #SAVE THE MODEL STATE DICTIONARY
# """this is going to be saved in the MODEL_SAVE_PATH and you dont have to use the obj= and the f= , they are the default"""
# torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)



#
# #RELOADING THE MODEL:
#
# #CREATE A NEW INSTANCE FOR THE MODEL
# loaded_model = modeloflinear()
# #LOAD THE SAVED DATA
# loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# #PUT THE LOADED_MODEL INTO THE DEVICE
# loaded_model.to(device)# we havent made the devinc thing yet because we are working on the cpu
#
# """ and by this you are going to have the same parameters loaded again to the model """
















# run prediction — we wrap this in inference_mode so PyTorch doesnt
        # waste memory tracking gradients (we are not training, just predicting)
        with torch.inference_mode():
            logits = model(img_tensor)            # raw numbers out of the model, not probabilities yet
                                                  # example: tensor([2.3, -1.1]) — hard to interpret directly

            probabilities = torch.softmax(logits, dim=1)  # converts raw numbers to probabilities that sum to 1
                                                           # example: tensor([0.91, 0.09]) — now readable
                                                           # dim=1 means we do this across the class dimension

            confidence, predicted_class = torch.max(probabilities, dim=1)
            # torch.max returns two things:
            # confidence = the highest probability value, example: 0.91
            # predicted_class = the index of that highest value, example: 0 (which means "closed")

        label = CLASS_NAMES[predicted_class.item()]   # converts index to human readable name
                                                       # 0 → "closed", 1 → "open"
                                                       # .item() converts tensor to a plain python number

        confidence_value = confidence.item()           # same, converts tensor to plain float

        print(f"[PREDICTION] {label} | confidence: {confidence_value:.2f}")
        # example output: [PREDICTION] open | confidence: 0.87
        # :.2f means show only 2 decimal places

        # only trigger the shortcut if:
        # 1. the label is "open" AND
        # 2. the model is confident enough (above your threshold)
        if label == "open" and confidence_value >= CONFIDENCE_THRESHOLD:
            if not shortcut_triggered:            # this check stops it from firing every second
                                                  # while the door stays open — fires only ONCE per open event
                keyboard.press_and_release(SHORTCUT)
                print(f"[ACTION] Door is open! Triggered shortcut: {SHORTCUT}")
                shortcut_triggered = True         # mark that we already triggered so we dont repeat

        else:
            shortcut_triggered = False            # door is closed again, reset the flag
                                                  # so next time it opens the shortcut fires again

    # build the text to show on the live video window
    # if label exists show it, otherwise show "warming up..." for the first second before first prediction
    display_label = f"{label} ({confidence_value:.2f})" if 'label' in locals() else "warming up..."

    cv2.putText(cropped, display_label, (10, 30),   # draw the text on the cropped door image
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,       # font style and size
                (0, 255, 0) if locals().get('label') == 'open' else (0, 0, 255),
                # green color if open, red color if closed
                2)                                   # thickness of the text

    cv2.imshow("Door Detection", cropped)            # show the window with the live door feed

    if cv2.waitKey(1) & 0xFF == ord('q'):            # check every 1ms if you pressed Q
        break                                        # if yes, exit the loop

cap.release()           # release the camera so other apps can use it
cv2.destroyAllWindows() # close all opencv windows cleanly
print("[INFO] Detection stopped.")