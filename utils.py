import json
import os
import cv2
import json
import shutil
import dtlpy as dl
from dtlpyconverters import services, coco_converters


def edit_ids(json_file, start_num):
    indicator = start_num
    with open(json_file, 'r') as f:
        data = json.load(f)

        for img in data["images"]:
            if img["id"] is not None:
                for annotation in data["annotations"]:
                    if annotation["image_id"] == img["id"]:
                        annotation["image_id"] = indicator
                img["id"] = indicator
                indicator += 1

        for ann in data["annotations"]:
            if ann["id"] is not None:
                ann["id"] = indicator
                indicator += 1

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def convert_json(input_file, ids_indicator, output_file):
    edit_ids(json_file=input_file, start_num=ids_indicator)
    with open(input_file, 'r') as f:
        data = json.load(f)

    images_dict = {}
    new_images = []
    new_annotations = []
    new_categories = data.get("categories", [])

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in images_dict:
            images_dict[image_id] = len(images_dict)
            image_info = data["images"][image_id]
            new_images.append({
                "id": images_dict[image_id],
                "width": image_info["width"],
                "height": image_info["height"],
                "file_name": image_info["file_name"],
                "license": 1
            })

        new_annotation = {
            "id": annotation["id"],
            "image_id": images_dict[image_id],
            "category_id": annotation["category_id"],
            "bbox": annotation["bbox"],
            "area": annotation["area"],
            "iscrowd": annotation["iscrowd"]
        }
        new_annotations.append(new_annotation)

    new_data = {
        "info": {
            "description": "My COCO dataset",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": "2022-01-01T00:00:00"
        },
        "licenses": [],
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)


def change_dataset_directories(model_entity: dl.Model, default_path=None):
    if default_path is None:
        default_path = os.path.join(os.getcwd(), 'tmp', model_entity.id, 'datasets', model_entity.dataset.id)

    new_path = os.path.join(os.getcwd(), 'datasets', model_entity.dataset.id)

    # Create directories if they don't exist
    for path in [new_path, os.path.join(new_path, 'train2017'), os.path.join(new_path, 'val2017'),
                 os.path.join(new_path, 'annotations')]:
        os.makedirs(path, exist_ok=True)

    # Move items to the new paths - move instead of copy to avoid re-downloading in the base model adapter
    shutil.copytree(src=os.path.join(default_path, 'train', 'items', 'train'),
                    dst=os.path.join(new_path, 'train2017'), dirs_exist_ok=True)

    shutil.copytree(src=os.path.join(default_path, 'validation', 'items', 'validation'),
                    dst=os.path.join(new_path, 'val2017'), dirs_exist_ok=True)

    return default_path, new_path


def convert_dataset(input_path, output_path, dataset):
    if not os.path.exists(os.path.join(input_path, 'coco.json')):
        conv = coco_converters.DataloopToCoco(output_annotations_path=output_path,
                                              input_annotations_path=input_path,
                                              download_items=False,
                                              download_annotations=False,
                                              dataset=dataset)
        coco_converter_services = services.converters_service.DataloopConverters()
        loop = coco_converter_services._get_event_loop()
        try:
            loop.run_until_complete(conv.convert_dataset())
        except Exception as e:
            raise e


def dtlpy_to_coco(input_path, output_path, dataset: dl.Dataset, ids_indicator=0):
    default_train_path = os.path.join(input_path, 'train', 'json')
    default_validation_path = os.path.join(input_path, 'validation', 'json')

    # Convert train and validations sets to coco format using dtlpy converters
    convert_dataset(input_path=default_train_path, output_path=default_train_path, dataset=dataset)
    convert_dataset(input_path=default_validation_path, output_path=default_validation_path, dataset=dataset)

    # Convert images and annotations to integers and move to the expected directory
    convert_json(input_file=os.path.join(default_train_path, 'coco.json'), ids_indicator=ids_indicator,
                 output_file=os.path.join(output_path, 'annotations', 'train_ann_coco.json'))
    convert_json(input_file=os.path.join(default_validation_path, 'coco.json'), ids_indicator=ids_indicator,
                 output_file=os.path.join(output_path, 'annotations', 'val_ann_coco.json'))


