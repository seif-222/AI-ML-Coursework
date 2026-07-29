import gradio as gr
import os
import torch

from model import create_eff_b2_model  # ✅ CHANGE TO THIS
from timeit import default_timer as timer
from typing import Tuple,Dict,List

#make the classes
classes = ["pizza","steak","sushi"]
#make the model and the transformer
eff_b2_model , eff_b2_model_transform = create_eff_b2_model(num_classes=len(classes))
#load the model
eff_b2_model.state_dict(torch.load(f="./09_EFFICIENTNETB2_MODEL_TRAINED_10EPOCHES_20%PERCENT.pth",map_location=torch.device("cpu")))
#make the prediction function
def predict(image):
    start_time = timer()
    img_transformed = eff_b2_model_transform(image).unsqueeze(0)

    eff_b2_model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(eff_b2_model(img_transformed), dim=1)

    #create pred probs dictionary
    pred_probs_dictionary = {classes[i]: float(pred_probs[0][i]) for i in range(len(classes))}

    end_time = timer()
    time_taken = end_time - start_time
    return pred_probs_dictionary,time_taken


### 4. Gradio app ###

# Create title, description and article
title = "FoodVision Mini 🍕🥩🍣"
description = "An [EfficientNetB2 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model to classify images as pizza, steak or sushi."
article = "Created at [09. PyTorch Model Deployment]."
#make example list
example_list = [["examples/" + example for example in os.listdir("examples")]]
#make the interface
demo = gr.Interface(fn=predict, # maps inputs to outputs
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)
#launch it
demo.launch()
