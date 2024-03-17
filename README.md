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

## Cloning

For instruction how to clone the pretrained model for prediction
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predicting)

## Training and Fine-tuning

For fine tuning on a custom dataset,
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset)

### Editing the configuration

To edit configurations via the platform, go to the YOLOX page in the Model Management and edit the json
file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

The basic configurations included are:

* ```epochs```: number of epochs to train the model (default: 10)
* ```batch_size```: batch size to be used during the training (default: 4)
* ```resume```: boolean, for resume training - loading latest ckpt (default: ```False```)
* ```logger```:  logger type: tensorboard or wandb (default: ```tensorboard```)
* ```cache```: For cacheing img to ['ram','disc','None'] (default ```None```)
* ```ckpt```: ckpt file for loading to finetune the model (default ```None```)

## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy it, so it can be used
for prediction.

