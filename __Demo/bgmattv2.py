from numpy.core.fromnumeric import squeeze
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms

device = torch.device('cuda')
precision = torch.float16

model = torch.jit.load(r'TorchScript\torchscript_resnet50_fp32.pth')
model.backbone_scale = 0.25
model.refine_mode = 'sampling'
model.refine_sample_pixels = 80_000

model = model.to(device)

src_path = r'input_src\src.jpg'
bgr_path = r'input_bg\bg.jpg'

src = Image.open(src_path)
src = ToTensor()(src).unsqueeze(0)
src = src.cuda()

bgr = Image.open(bgr_path)
bgr = ToTensor()(bgr).unsqueeze(0)
bgr = bgr.cuda()

pha, fgr = model(src, bgr)[:2]
tgt_bgr = torch.tensor([120/255, 255/255, 155/255], device=device).view(1, 3, 1, 1)
com = fgr * pha + tgt_bgr * (1 - pha)

unloader = transforms.ToPILImage()

def image_loader(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

com = image_loader(com)
com.show()
