

from pathlib import Path

from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
)
from ultralytics.utils import ROOT, YAML


class YOLO(Model):
    

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:

            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        
        super().__init__(model=model, task="detect", verbose=verbose)


        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        
        self.model.set_classes(classes)

        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes


        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    

    def __init__(self, model="yoloe-11s-seg.pt", task=None, verbose=False) -> None:
        
        super().__init__(model=model, task=task, verbose=verbose)


        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab, names):
        
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes, embeddings):
        
        assert isinstance(self.model, YOLOEModel)
        self.model.set_classes(classes, embeddings)

        assert " " not in classes
        self.model.names = classes


        if self.predictor:
            self.predictor.model.names = classes

    def val(
        self,
        validator=None,
        load_vp=False,
        refer_data=None,
        **kwargs,
    ):
        
        custom = {"rect": not load_vp}
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict = {},
        refer_image=None,
        predictor=None,
        **kwargs,
    ):
        
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
        self.predictor = (predictor or self._smart_load("predictor"))(
            overrides={
                "task": self.model.task,
                "mode": "predict",
                "save": False,
                "verbose": refer_image is None,
                "batch": 1,
            },
            _callbacks=self.callbacks,
        )

        if len(visual_prompts):
            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list)
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())

        self.predictor.setup_model(model=self.model)

        if refer_image is None and source is not None:
            dataset = load_inference_source(source)
            if dataset.mode in {"video", "stream"}:

                refer_image = next(iter(dataset))[1][0]
        if refer_image is not None and len(visual_prompts):
            vpe = self.predictor.get_vpe(refer_image)
            self.model.set_classes(self.model.names, vpe)
            self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
            self.predictor = None

        return super().predict(source, stream, **kwargs)
