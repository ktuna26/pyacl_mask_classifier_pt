# PyTorch Face Mask Classifier
Please open the `jupyter-notebook` for a quick demo | [Pretrained Model](https://onebox.huawei.com/p/f966671c062074cd3b62d72dd5652b0f)

## Overview
In this project the `mobilenetv2` model is used to classify face images with/without mask.

## Getting started
Install dependencies;
- opencv-python>=3.4.2
- Pillow
- numpy

```
pip install -r requirements.txt
```
And then download the `PT` file of from the link.

### PT model -> ONNX format -> Ascend om format
#### PT -> ONNX
Use in the original repository  the `model/onnx_export.py` the script to convert `PT` file to `ONNX ` file.

#### ONNX -> OM
And then use in the same directory atc tool to convert `ONNX ` file to `OM` file as as follows.
```bash
atc --model=mask_model.onnx \
    --framework=5 \
    --output=mask_model \ 
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

Finaly, open `jupyter-notebook` and run the code for demo