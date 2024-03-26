import logging
import warnings
import torch.backends.cudnn as cudnn
import random
import cv2
import urllib.request
from glob import glob

from yolox.data import ValTransform, TrainTransform
from yolox.utils import configure_nccl, configure_omp

from utils import *
from exps import *

logger = logging.getLogger('ModelAdapter')


@dl.Package.decorators.module(description='Model Adapter for Yolovx object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):

    def __init__(self, model_entity: dl.Model):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cls_names = model_entity.labels
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
                    preproc=TrainTransform(
                        max_labels=50,
                        flip_prob=self.flip_prob,
                        hsv_prob=self.hsv_prob
                    ),
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
                urllib.request.urlretrieve(checkpoint_url,
                                           os.path.join(local_path, self.configuration.get('weights_filename')))
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
                    shutil.move(src_file, os.path.join(data_path, dst, os.path.basename(src_file)))
        for root, dirs, files in os.walk(data_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

    def convert_from_dtlpy(self, data_path, **kwargs):
        # Subsets validation
        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError('Could not find train set. Yolo-x requires train and validation set for training. '
                             'Add a train set DQL filter in the dl.Model metadata')
        if 'validation' not in subsets:
            raise ValueError('Could not find validation set. Yolo-x requires train and validation set for training. '
                             'Add a validation set DQL filter in the dl.Model metadata')

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            filters.add_join(field='type', values='box')
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(f'Could not find box annotations in subset {subset}. '
                                 f'Cannot train without annotation in the data subsets')

        self.move_annotation_files(os.path.join(data_path, 'train'))
        self.move_annotation_files(os.path.join(data_path, 'validation'))

        # Set Dataset directories as yolox requires
        default_path, new_data_path = change_dataset_directories(model_entity=self.model_entity)

        # Convert Train and Validation to coco format
        dtlpy_to_coco(input_path=default_path, output_path=new_data_path, dataset=self.model_entity.dataset)

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
        args = {'batch_size': self.configuration.get("batch_size", 4),
                'resume': self.configuration.get("resume", False),  # For resume training - loading latest ckpt
                'fp16': self.configuration.get("fp16", False),
                'occupy': self.configuration.get("occupy", False),  # True for pre-allocate gpu memory for training
                'logger': self.configuration.get("logger", 'tensorboard'),  # ['tensorboard','wandb']
                'cache': self.configuration.get("cache", None),  # For cacheing img to ['ram','disc','None']
                'ckpt': self.configuration.get("ckpt", None),  # ckpt file for loading to finetune the model
                'experiment_name': self.exp.exp_name}

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
        self.trainer = self.exp.get_trainer(args)

        logger.info(f"Trainer device: {self.trainer.device}, Trainer local rank: {self.trainer.local_rank}")
        logger.info(f"Trainer is_distributed: {self.trainer.is_distributed}, Trainer rank: {self.trainer.rank}")
        logger.info(f"TORCH: {torch.__version__}")

        self.trainer.train()

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = list()
        for img in batch:
            detections = self.inference(img)
            collection = dl.AnnotationCollection()
            if detections is not None:
                for detection in detections:
                    x0, y0, x1, y1, label, confidence = (detection["x0"], detection["y0"], detection["x1"],
                                                         detection["y1"], detection["label"],
                                                         detection["conf"])

                    collection.add(annotation_definition=dl.Box(left=max(x0, 0),
                                                                top=max(y0, 0),
                                                                right=min(x1, img.shape[1]),
                                                                bottom=min(y1, img.shape[0]),
                                                                label=label
                                                                ),

                                   model_info={'name': self.model_entity.name,
                                               'model_id': self.model_entity.id,
                                               'confidence': float(confidence)})

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
            outputs = postprocess(
                outputs, num_classes, self.exp.test_conf,
                self.exp.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        if outputs[0] is None:
            logger.warning(
                "Model's predictions confidence is less than confidence threshold - therefore no annotations found! ")
            return None
        else:
            bboxes, scores, cls = self.inference_results(output=outputs[0], img_info=img_info)

            box_annotations = self.create_box_annotations(boxes=bboxes,
                                                          scores=scores,
                                                          cls_ids=cls,
                                                          cls_names=self.model_entity.id_to_label_map)

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


def package_creation(project_id: str, package_name: str):
    code_base_path = os.path.abspath(
        os.path.join(os.getcwd(), os.pardir))
    project = dl.projects.get(project_id=project_id)
    codebase = project.codebases.pack(directory=os.getcwd())
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={'weights_filename': 'yolox_s.pth',
                                                                 'exp_class_name': 'SmallExp',
                                                                 'epochs': 10,
                                                                 'batch_size': 4,
                                                                 'conf_thres': 0.25},
                                          output_type=dl.AnnotationType.BOX
                                          )

    modules = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name=package_name,
                                    src_path=os.getcwd(),
                                    package_type='ml',
                                    codebase=codebase,
                                    modules=[modules],
                                    is_global=False,
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_S,
                                                                        # runnerImage=
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)
    return package


def create_model(package: dl.Package, model_name, artifact_path, labels, dataset_id: str):
    try:
        model = package.models.create(model_name=model_name,
                                      dataset_id=dataset_id,
                                      description='yolox pretrain - coco dataset',
                                      tags=['yolox', 'pretrained', "coco-ms"],
                                      status='trained',
                                      labels=labels,
                                      configuration={'weights_filename': 'yolox_s.pth',
                                                     'exp_class_name': 'SmallExp',
                                                     'epochs': 10,
                                                     'batch_size': 4,
                                                     'conf_thres': 0.25},
                                      project_id=package.project.id,
                                      )

    except dl.exceptions.BadRequest:
        model = package.models.get(model_name=model_name)

    artifact = model.artifacts.upload(filepath=artifact_path)
    model.configuration['weights_filename'] = artifact.filename

    if model.status == 'deployed':
        for service_id in model.metadata.get('system', dict()).get('deploy', dict()).get('services', list()):
            service = dl.services.get(service_id=service_id)
            service.update()
    else:
        model.deploy(service_config=package.service_config)
    return model
