# YOLOX Object Detection Model Adapter

## Introduction

This repo is a model integration between [MegEngine implementation of PyTorch YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) object 
detection model and [Dataloop](https://dataloop.ai/).

YOLOX is another improved version of YOLO, with a simpler design but better performance. It aims to bridge the gap 
between research and industrial communities. By leveraging advanced techniques such as anchor-free detection and dynamic
head construction, YOLOX achieves high performance while maintaining high efficiency.

## Requirements

* yolox
* dtlpy
* dtlpy-converters
* An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the package and create the YOLOv5 model adapter, you will need
a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and
a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform. The dataset should
have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory)
containing its training and validation subsets.

## Available Models

YOLOX comes in several variants to accommodate different performance and resource requirements:

- **YOLOX-Tiny**: Smallest model, ideal for resource-constrained environments
- **YOLOX-S**: Small model with good balance of performance and speed
- **YOLOX-M**: Medium-sized model with improved accuracy
- **YOLOX-L**: Large model with higher accuracy
- **YOLOX-XL**: Extra large model with the highest accuracy

All models are pre-trained on the COCO dataset and can detect 80 different object classes.

## Cloning

For instruction how to clone the pretrained model for prediction
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predicting)

## Training and Fine-tuning

For fine tuning on a custom dataset,
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset)
Please note that the model should be trained using GPU.

### Editing the configuration

To edit configurations via the platform, go to the YOLOX page in the Model Management and edit the json
file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

The basic configurations included are:

* ```weights_filename```: Name of the weights file (e.g., "yolox_tiny.pth")
* ```checkpoint_url```: URL to download the model weights
* ```exp_class_name```: Experiment class name (TinyExp, SmallExp, MediumExp, LargeExp, XLargeExp)
* ```epochs```: Number of epochs to train the model (default: 10)
* ```batch_size```: Batch size to be used during the training (default: 4)
* ```conf_thres```: Confidence threshold for detections (default: 0.25)
* ```resume```: Boolean, for resume training - loading latest checkpoint (default: ```False```)
* ```fp16```: Enable mixed precision training (default: ```False```)
* ```occupy```: Preemptive occupation of GPU memory (default: ```False```)
* ```logger```: Logger type: tensorboard or wandb (default: ```tensorboard```)
* ```cache```: For caching images to ['ram','disc','None'] (default: ```None```)
* ```ckpt```: Checkpoint file for loading to finetune the model (default: ```None```)

## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy it, so it can be used
for prediction.

