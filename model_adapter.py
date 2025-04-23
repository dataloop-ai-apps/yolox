import logging
import warnings
import torch.backends.cudnn as cudnn
import random
import cv2
import urllib.request
from glob import glob
import numpy as np

from yolox.data import ValTransform, TrainTransform
from yolox.utils import configure_nccl, configure_omp

from utils import *
from exps import *

logger = logging.getLogger('ModelAdapter')


class ModelAdapter(dl.BaseModelAdapter):

    def __init__(self, model_entity: dl.Model = None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.exp = None
        self.trainer = None

        super().__init__(model_entity)

    def initialize_experiment(self, class_name: str):
        creator = DynamicClassCreator(class_name=class_name)

        # Access the dynamically created class
        DynamicClass = getattr(creator, class_name)

        class DtlExp(DynamicClass):
            def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
                return DtlDataset(
                    data_dir=self.data_dir,
                    json_file=self.train_ann,
                    img_size=self.input_size,
                    preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                    cache=cache,
                    cache_type=cache_type,
                )

            def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
                return DtlEvaluator(
                    dataloader=self.get_eval_loader(batch_size, is_distributed, testdev=testdev, legacy=legacy),
                    img_size=self.test_size,
                    confthre=self.test_conf,
                    nmsthre=self.nmsthre,
                    num_classes=self.num_classes,
                    testdev=testdev,
                )

        return DtlExp()

    def load(self, local_path, **kwargs):
        model_filename = os.path.join(local_path, self.configuration.get('weights_filename', 'weights/best_ckpt.pth'))
        checkpoint_url = self.configuration.get('checkpoint_url', None)

        # Experiment initialization
        self.exp = self.initialize_experiment(self.model_entity.configuration.get("exp_class_name", "SmallExp"))

        self.model = self.exp.get_model()
        self.model.eval()

        # Load weights from url
        if not os.path.isfile(model_filename):
            if checkpoint_url is not None:
                logger.info("Loading weights from url")
                os.makedirs(local_path, exist_ok=True)
                logger.info("created local_path dir")
                urllib.request.urlretrieve(
                    checkpoint_url, os.path.join(local_path, self.configuration.get('weights_filename'))
                )
            else:
                raise Exception("checkpoints weights were not loaded! URL not found")

        if os.path.exists(model_filename):
            logger.info("Loading saved weights")
            weights = torch.load(model_filename, map_location=self.device)
            self.model.load_state_dict(weights["model"])
        else:
            raise dl.exceptions.NotFound(f'Model path ({model_filename}) not found! model weights is required')

    @staticmethod
    def move_annotation_files(data_path):
        json_files = glob(os.path.join(data_path, 'json', '**/*.json'))
        json_files += glob(os.path.join(data_path, 'json', '*.json'))

        if os.path.sep == '\\':
            sub_path = '\\'.join(json_files[0].split('json\\')[-1].split('\\')[:-1])
        else:
            sub_path = '/'.join(json_files[0].split('json/')[-1].split('/')[:-1])

        item_files = glob(os.path.join(data_path, 'items', sub_path, '*'))

        for src, dst in zip([json_files, item_files], ['json', 'items']):
            for src_file in src:
                if not os.path.exists(os.path.join(data_path, dst, os.path.basename(src_file))):
                    shutil.move(src_file, os.path.join(data_path, dst, sub_path, os.path.basename(src_file)))
        for root, dirs, files in os.walk(data_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

    def convert_from_dtlpy(self, data_path, **kwargs):

        self.move_annotation_files(os.path.join(data_path, 'train'))
        self.move_annotation_files(os.path.join(data_path, 'validation'))

        # Set Dataset directories as yolox requires
        default_path, new_data_path = change_dataset_directories(model_entity=self.model_entity)

        # Convert Train and Validation to coco format
        dtlpy_to_coco(
            input_path=default_path,
            output_path=new_data_path,
            dataset=self.model_entity.dataset,
            label_to_id_mapping=self.model_entity.label_to_id_map,
        )

        self.exp.train_ann = 'train_ann_coco.json'
        self.exp.val_ann = 'val_ann_coco.json'
        self.exp.data_dir = new_data_path

    def save(self, local_path, **kwargs):

        self.trainer.file_name = os.path.join(local_path, 'weights')
        self.trainer.save_ckpt(ckpt_name="best")
        self.configuration.update({'weights_filename': 'weights/best_ckpt.pth'})

        logger.info(f"Saved state dict at {os.path.join(self.trainer.file_name, 'best_ckpt.pth')}")

    def train(self, data_path, output_path, **kwargs):
        # Creating args class for passing config for the experiment's trainer
        class Args:
            def __init__(self, **entries):
                self.__dict__.update(entries)

        # Reading config - for trainer
        args = {
            'batch_size': self.configuration.get("batch_size", 4),
            'resume': self.configuration.get("resume", False),  # For resume training - loading latest ckpt
            'fp16': self.configuration.get("fp16", False),
            'occupy': self.configuration.get("occupy", False),  # True for pre-allocate gpu memory for training
            'logger': self.configuration.get("logger", 'tensorboard'),  # ['tensorboard','wandb']
            'cache': self.configuration.get("cache", None),  # For cacheing img to ['ram','disc','None']
            'ckpt': self.configuration.get("ckpt", None),  # ckpt file for loading to finetune the model
            'experiment_name': self.exp.exp_name,
        }

        # Seed training
        if self.exp.seed is not None:
            random.seed(self.exp.seed)
            torch.manual_seed(self.exp.seed)
            cudnn.deterministic = True
            warnings.warn(
                "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! You may see unexpected behavior "
                "when restarting from checkpoints."
            )

        # set environment variables for distributed training
        configure_nccl()
        configure_omp()
        cudnn.benchmark = True

        args = Args(**args)
        self.exp.num_classes = self.configuration.get("num_classes", len(self.model_entity.labels))
        self.exp.max_epoch = self.configuration.get("epoch", 10)

        # Store the original after_epoch method before initializing the trainer
        original_after_epoch = self.exp.get_trainer(args).after_epoch
        
        # Create a custom trainer class that extends the original one
        class CustomTrainer(type(self.exp.get_trainer(args))):
            def after_epoch(custom_self):
                # Call the original after_epoch method
                original_after_epoch(custom_self)
                
                # Your custom metrics logic
                current_epoch = custom_self.epoch + 1
                logger.info(f"Custom metrics collection for epoch {current_epoch}")
                
                if custom_self.rank == 0:
                    metrics = {}
                    if hasattr(custom_self, 'evaluate_results') and custom_self.evaluate_results is not None:
                        eval_results = custom_self.evaluate_results
                        if 'COCO_AP' in eval_results:
                            metrics['val/mAP50-95(B)'] = eval_results['COCO_AP']
                        if 'COCO_AP50' in eval_results:
                            metrics['val/mAP50(B)'] = eval_results['COCO_AP50']
                        if 'COCO_AP75' in eval_results:
                            metrics['val/mAP75'] = eval_results['COCO_AP75']

                    if hasattr(custom_self, 'loss_meter'):
                        metrics['train/box_loss'] = custom_self.loss_meter.avg_statistics.get('loss_box', 0)
                        metrics['train/cls_loss'] = custom_self.loss_meter.avg_statistics.get('loss_cls', 0)
                        metrics['train/obj_loss'] = custom_self.loss_meter.avg_statistics.get('loss_obj', 0)
                        metrics['train/total_loss'] = custom_self.loss_meter.avg_statistics.get('loss', 0)

                    samples = []
                    NaN_dict = {
                        'box_loss': 1,
                        'cls_loss': 1,
                        'obj_loss': 1,
                        'total_loss': 1,
                        'mAP50(B)': 0,
                        'mAP50-95(B)': 0,
                        'mAP75': 0,
                    }

                    for metric_name, value in metrics.items():
                        legend, figure = metric_name.split('/')
                        logger.info(f'Updating figure {figure} with legend {legend} with value {value}')

                        if not np.isfinite(value):
                            filters = dl.Filters(resource=dl.FiltersResource.METRICS)
                            filters.add(field='modelId', values=self.model_entity.id)
                            filters.add(field='figure', values=figure)
                            filters.add(field='data.x', values=current_epoch - 1)
                            items = self.model_entity.metrics.list(filters=filters)

                            if items.items_count > 0:
                                value = items.items[0].y
                            else:
                                value = NaN_dict.get(figure, 0)
                            logger.warning(f'Value is not finite. For figure {figure} and legend {legend} using value {value}')

                        samples.append(dl.PlotSample(figure=figure,
                                                    legend=legend,
                                                    x=current_epoch,
                                                    y=value))

                    if samples:
                        self.model_entity.metrics.create(samples=samples,
                                                        dataset_id=self.model_entity.dataset_id)
                    self.configuration['current_epoch'] = current_epoch
                    self.save_to_model(local_path=output_path, cleanup=False)

        # Override the get_trainer method in your experiment to return our custom trainer
        original_get_trainer = self.exp.get_trainer
        def custom_get_trainer(args):
            return CustomTrainer(self.exp, args)
        
        self.exp.get_trainer = custom_get_trainer
        
        # Now create the trainer with our custom class
        self.trainer = self.exp.get_trainer(args)
        
        logger.info("Custom trainer with metrics collection has been set up")

        try:
            self.trainer.train()
            logger.info("Training finished successfully.")
        except Exception as e:
            logger.error(f"Error during self.trainer.train(): {e}", exc_info=True)
            raise e

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = list()
        for img in batch:
            detections = self.inference(img)
            collection = dl.AnnotationCollection()
            if detections is not None:
                for detection in detections:
                    x0, y0, x1, y1, label, confidence = (
                        detection["x0"],
                        detection["y0"],
                        detection["x1"],
                        detection["y1"],
                        detection["label"],
                        detection["conf"],
                    )

                    collection.add(
                        annotation_definition=dl.Box(
                            left=max(x0, 0),
                            top=max(y0, 0),
                            right=min(x1, img.shape[1]),
                            bottom=min(y1, img.shape[0]),
                            label=label,
                        ),
                        model_info={
                            'name': self.model_entity.name,
                            'model_id': self.model_entity.id,
                            'confidence': float(confidence),
                        },
                    )

                batch_annotations.append(collection)

        return batch_annotations

    def inference(self, img):
        # Predict related config's parameters
        fp16 = self.configuration.get("fp16", False)
        decoder = self.configuration.get("decoder", None)

        img_info = {"raw_img": img}
        ratio = min(self.exp.test_size[0] / img.shape[0], self.exp.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        preproc = ValTransform(legacy=None)
        img, _ = preproc(img, None, self.exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            # MIXED PRECISION EVALUATION - match each operation to its appropriate datatype(torch.float32/16)
            if fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if decoder is not None:
                outputs = decoder(outputs, dtype=outputs.type())

            try:
                num_classes = len(self.model_entity.dataset.labels)
            except RuntimeError:
                num_classes = len(self.model_entity.id_to_label_map.values())

            # NMS - POST PROCESSING STEP
            outputs = postprocess(outputs, num_classes, self.exp.test_conf, self.exp.nmsthre, class_agnostic=True)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        if outputs[0] is None:
            logger.warning(
                "Model's predictions confidence is less than confidence threshold - therefore no annotations found! "
            )
            return None
        else:
            bboxes, scores, cls = self.inference_results(output=outputs[0], img_info=img_info)

            box_annotations = self.create_box_annotations(
                boxes=bboxes, scores=scores, cls_ids=cls, cls_names=self.model_entity.id_to_label_map
            )

        return box_annotations

    @staticmethod
    def inference_results(output, img_info):
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= img_info["ratio"]
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        return bboxes, scores, cls

    @staticmethod
    def create_box_annotations(boxes, scores, cls_ids, cls_names, conf=0.5):
        box_annotations = []
        for i in range(len(boxes)):
            i_box = {"id": i}
            box = boxes[i]
            i_box["label"] = cls_names[int(cls_ids[i])]
            score = scores[i]
            if score < conf:
                continue
            i_box["x0"] = int(box[0])
            i_box["y0"] = int(box[1])
            i_box["x1"] = int(box[2])
            i_box["y1"] = int(box[3])
            i_box["conf"] = score.item()

            box_annotations.append(i_box)

        return box_annotations

    def prepare_item_func(self, item: dl.Item):
        path = item.download(save_locally=True)
        if os.path.exists(path):
            img = cv2.imread(path)
        else:
            img = None
        os.remove(path)
        return img