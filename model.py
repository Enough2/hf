import torch
import torchvision.transforms as transforms
import torchvision.models as models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import time

class Prediction:
    def __init__(self, data, heatmap, duration):
        self.data = data
        self.heatmap = heatmap
        self.duration = duration

class Pipeline:
    def __init__(self):
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = {}

        resnet50_0810 = self.to_device(ResNet50(self.classes), self.device)
        resnet50_0810.load_state_dict(torch.load('models/resnet50_0810.pt', map_location=self.device))
        resnet50_0810.eval()
        resnet50_0810.cam = GradCAM(resnet50_0810.network, [resnet50_0810.network.layer4], torch.cuda.is_available())
        self.model["resnet50_0810"] = resnet50_0810

        resnet152_0813 = self.to_device(ResNet152(self.classes), self.device)
        resnet152_0813.load_state_dict(torch.load('models/resnet152_0813.pt', map_location=self.device))
        resnet152_0813.eval()
        resnet152_0813.cam = GradCAM(resnet152_0813.network, [resnet152_0813.network.layer4], torch.cuda.is_available())
        self.model["resnet152_0813"] = resnet152_0813

        resnet152_0902 = self.to_device(ResNet152(self.classes), self.device)
        resnet152_0902.load_state_dict(torch.load('models/resnet152_0902.pt', map_location=self.device))
        resnet152_0902.eval()
        resnet152_0902.cam = GradCAM(resnet152_0902.network, [resnet152_0902.network.layer4], torch.cuda.is_available())
        self.model["resnet152_0902"] = resnet152_0902

    def to_device(self, data, device):
        return data.to(device, torch.float32)

    def predict_image(self, model, image):
        tensor = self.transformations(image)
        xb = self.to_device(tensor.unsqueeze(0), self.device)
        start_time = time.time()
        yb = self.model[model](xb)
        end_time = time.time()
        data = {self.classes[i]: float(yb[0][i]) for i in range(len(self.classes))}
        return Prediction(data, self.visualize(model, image, xb), int((end_time - start_time) * 1000))

    def visualize(self, model, rgb_image, input_tensor):
        rgb_image = rgb_image.resize((256, 256))
        rgb_image = np.array(rgb_image)
        rgb_image = np.float32(rgb_image) / 255
        greyscale_cam = self.model[model].cam(input_tensor)[0, :]
        image = show_cam_on_image(rgb_image, greyscale_cam, use_rgb=True)
        return Image.fromarray(image)

class ResNet50(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.network = models.resnet50(weights="DEFAULT")
        self.network.fc = torch.nn.Linear(self.network.fc.in_features, len(classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

class ResNet152(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.network = models.resnet152(weights="DEFAULT")
        self.network.fc = torch.nn.Linear(self.network.fc.in_features, len(classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))