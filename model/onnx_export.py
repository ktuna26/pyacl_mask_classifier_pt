import torch
from model import MaskClassifier
#from PIL import Image

# load model
model = MaskClassifier().to('cpu')

model.load_state_dict(torch.load('mask_model.pt', map_location='cpu')['state_dict'])
model.eval()

# load data
img = torch.zeros((1, 3, 128, 128)) # BCHW
#img = model.preprocess(Image.open("no_mask.jpg")).unsqueeze(0).to('cpu')

# trace export
torch.onnx.export(model, img, 'mask_model.onnx', export_params=True, verbose=True, opset_version=11)