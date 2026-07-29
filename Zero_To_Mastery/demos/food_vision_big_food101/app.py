#making the app.py file
### imports
import gradio as gr
import os
import torch

from model import create_eff_b2_model# no problem if this import was red in the notebook because here we are dealing as we are in the demos/DEMO_NAME directory
from timeit import default_timer as timer
from typing import Tuple, Dict

#set up classes names
with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()] #what the strip does is that it removes the "\n" that is there to make a new line for every class

#make the model and the transformer
eff_b2_model , eff_b2_transformer = create_eff_b2_model(num_classes=101)
#laod the saved model
eff_b2_model.load_state_dict(torch.load("FINAL_MODEL_EFF_B2.pth",
                                        map_location=torch.device('cpu')))#load on cpu , because users mostly have cpu only
### predict funciton
def predict(img):
    start_time = timer()
    #
    img=eff_b2_transformer(img).unsqueeze(dim=0)
    #
    eff_b2_model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(eff_b2_model(img), dim=1)
    #
    pred_labels_and_probs = {class_names[i]:float(pred_probs[0][i]) for i in range(len(class_names))} # remember has to be float for gradio
    #
    end_time=timer()
    time_taken = round(end_time-start_time,2)
    #
    return pred_labels_and_probs, time_taken


### 4. Gradio app ###

# Create title, description and article
title = "FoodVision BIG 🍔👁💪"
description = "based on 101 classes"
article = "Created with EfficientNetB2"
# Create example list
example_list = [
    ["examples/1011328.jpg"],
    ["examples/753322.jpg"],
    ["examples/2228627.jpg"]
]# that is becasue of some error that os.listdir makes so we made it manually

# Create the Gradio demo
demo = gr.Interface(fn=predict, # maps inputs to outputs
                  inputs=gr.Image(type="pil"),
                  outputs=[gr.Label(num_top_classes=5, label="Predictions"),
                          gr.Number(label="Prediction time (s)")],
                  examples=example_list,
                  title=title,
                  description=description,
                  article=article)

# Launch the demo!
demo.launch()
