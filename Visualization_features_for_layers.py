################################################## KE LIANG ############################################################
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
from config import _C as cfg
from model import get_model

model = get_model(model_name=cfg.MODEL.ARCH)
resume_path = 'checkpoint/Modified_Residual_Transfer_Model_apparent_2.pth'##modify the path of the model here
# resume_path = 'checkpoint/Modified_Residual_Transfer_Model_real_2.pth'
checkpoint = torch.load(resume_path, map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])
image = Image.open('visual_layers/000096.jpg') ##modify the path of the model here

transform = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
img = transform(image)
img = img.unsqueeze(0)

def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')



new_model = nn.Sequential(*list(model.children())[:0])
f1 = new_model(img)
# save_img(f1, 'visual_layers/Real/layer1')
save_img(f1, 'visual_layers/Apparent/layer1')##modify the path of output

new_model = nn.Sequential(*list(model.children())[:1])
f2 = new_model(img)
# save_img(f2, 'visual_layers/Real/layer2')
save_img(f2, 'visual_layers/Apparent/layer2')##modify the path of output

new_model = nn.Sequential(*list(model.children())[:2])
f3 = new_model(img)
# save_img(f3, 'visual_layers/Real/layer3')
save_img(f3, 'visual_layers/Apparent/layer3')##modify the path of output

new_model = nn.Sequential(*list(model.children())[:3])
f4 = new_model(img)
# save_img(f4, 'visual_layers/Real/layer4')
save_img(f4, 'visual_layers/Apparent/layer4')##modify the path of output

new_model = nn.Sequential(*list(model.children())[:4])
f5 = new_model(img)
# save_img(f5, 'visual_layers/Real/layer5')
save_img(f5, 'visual_layers/Apparent/layer5')##modify the path of output

new_model = nn.Sequential(*list(model.children())[:5])
f6 = new_model(img)
# save_img(f6, 'visual_layers/Real/layer6')
save_img(f6, 'visual_layers/Apparent/layer6')##modify the path of output

new_model = nn.Sequential(*list(model.children())[:6])
f7 = new_model(img)
print(f7.shape)
# save_img(f7, 'visual_layers/Real/layer7')
save_img(f7, 'visual_layers/Apparent/layer7')##modify the path of output

new_model = nn.Sequential(*list(model.children())[:7])
f8 = new_model(img)
print(f8.shape)
# save_img(f8, 'visual_layers/Real/layer8')
save_img(f7, 'visual_layers/Apparent/layer8')##modify the path of output

new_model = nn.Sequential(*list(model.children())[:8])
f9 = new_model(img)
print(f9.shape)
# save_img(f9, 'visual_layers/Real/layer9')
save_img(f9, 'visual_layers/Apparent/layer9')##modify the path of output
########################################################################################################################
