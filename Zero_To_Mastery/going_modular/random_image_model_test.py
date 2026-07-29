#making a fuction to seee how much the model cna accually guess in random images
from torchvision import transforms
import torch
from PIL import Image
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def random_image_model_test (model,
                             image_path,
                             class_names,
                             image_size=224,
                             transform=None,
                             device=device):
    #load image
    img = Image.open(image_path)
    #make transform
    if transform == None :
        image_transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        image_transform = transform
    #model to device and......
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        img_model_pred = model(image_transform(img).unsqueeze(0).to(device))
    img_model_pred_prob = torch.softmax(img_model_pred, dim=1)
    img_model_pred_label = torch.argmax(img_model_pred_prob, dim=1)
    #make a figure
    plt.figure()

    plt.imshow(img)
    plt.title(f"Pred: {class_names[img_model_pred_label]} | Prob: {img_model_pred_prob.max():.3f}")
    plt.axis(False);


