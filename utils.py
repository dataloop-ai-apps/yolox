import os
import json
import shutil
import asyncio
import dtlpy as dl
from dtlpyconverters import services, coco_converters
from PIL import Image

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


def convert_json(input_file, ids_indicator, output_file, filename_mapping, data_dir, name):
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
            original_name = image_info["file_name"]
            if original_name in filename_mapping:
                new_filename = filename_mapping[original_name]
            else:
                raise ValueError(f"Could not find mapping for file: {original_name}")
            if image_info["width"] is None or image_info["height"] is None:
                im = Image.open(os.path.join(data_dir, name, new_filename))
                image_info["width"] = im.width
                image_info["height"] = im.height
            new_images.append(
                {
                    "id": images_dict[image_id],
                    "width": image_info["width"],
                    "height": image_info["height"],
                    "file_name": new_filename,
                    "license": 1,
                }
            )

        new_annotation = {
            "id": annotation["id"],
            "image_id": images_dict[image_id],
            "category_id": annotation["category_id"],
            "bbox": annotation["bbox"],
            "area": annotation["area"],
            "iscrowd": annotation["iscrowd"],
        }
        new_annotations.append(new_annotation)

    new_data = {
        "info": {"description": "COCO format dataset"},
        "licenses": [],
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories,
    }

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)


def change_dataset_directories(new_path, model_entity: dl.Model, default_path=None):
    # Create directories if they don't exist
    for path in [
        new_path,
        os.path.join(new_path, 'train2017'),
        os.path.join(new_path, 'val2017'),
        os.path.join(new_path, 'annotations'),
    ]:
        os.makedirs(path, exist_ok=True)

    # Move items to the new paths - move instead of copy to avoid re-downloading in the base model adapter
    for item in os.listdir(os.path.join(default_path, 'train', 'items')):
        shutil.move(
            src=os.path.join(default_path, 'train', 'items', item),
            dst=os.path.join(new_path, 'train2017', item)
        )

    for item in os.listdir(os.path.join(default_path, 'validation', 'items')):
        shutil.move(
            src=os.path.join(default_path, 'validation', 'items', item),
            dst=os.path.join(new_path, 'val2017', item)
        )



def convert_dataset(input_path, output_path, dataset, label_to_id_mapping):
    if not os.path.exists(os.path.join(input_path, 'coco.json')):

        conv = coco_converters.DataloopToCoco(
            output_annotations_path=output_path,
            input_annotations_path=input_path,
            download_items=False,
            download_annotations=False,
            label_to_id_mapping=label_to_id_mapping,
            dataset=dataset,
        )
        asyncio.run(conv.convert_dataset())


def dtlpy_to_coco(input_path, 
                  output_path, 
                  dataset: dl.Dataset,
                  label_to_id_mapping: dict,
                  ids_indicator=0,
                  filename_mapping=None):
    default_train_path = os.path.join(input_path, 'train', 'json')
    default_validation_path = os.path.join(input_path, 'validation', 'json')

    # Convert train and validations sets to coco format using dtlpy converters
    convert_dataset(
        input_path=default_train_path,
        output_path=default_train_path,
        dataset=dataset,
        label_to_id_mapping=label_to_id_mapping,
    )
    convert_dataset(
        input_path=default_validation_path,
        output_path=default_validation_path,
        dataset=dataset,
        label_to_id_mapping=label_to_id_mapping,
    )

    # Convert images and annotations to integers and move to the expected directory
    convert_json(
        input_file=os.path.join(default_train_path, 'coco.json'),
        ids_indicator=ids_indicator,
        output_file=os.path.join(output_path, 'annotations', 'train_ann_coco.json'),
        filename_mapping=filename_mapping,
        data_dir=output_path,
        name='train2017',
    )
    convert_json(
        input_file=os.path.join(default_validation_path, 'coco.json'),
        ids_indicator=ids_indicator,
        output_file=os.path.join(output_path, 'annotations', 'val_ann_coco.json'),
        filename_mapping=filename_mapping,
        data_dir=output_path,
        name='val2017',
    )
