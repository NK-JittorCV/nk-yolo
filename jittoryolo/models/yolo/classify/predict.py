import cv2
import jittor as jt
from PIL import Image

from jittoryolo.engine.predictor import BasePredictor
from jittoryolo.engine.results import Results
from jittoryolo.utils import DEFAULT_CFG, ops


class ClassificationPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from jittoryolo.utils import ASSETS
        from jittoryolo.models.yolo.classify import ClassificationPredictor

        args = dict(model="yolov8n-cls.pt", source=ASSETS)
        predictor = ClassificationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes ClassificationPredictor setting the task to 'classify'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"
        self._legacy_transform_name = "jittoryolo.yolo.data.augment.ToTensor"

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, jt.Var):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:  # to handle legacy transforms
                img = jt.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = jt.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                )
        img = (img if isinstance(img, jt.Var) else jt.array(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(orig_imgs, list):  # input images are a jt.Var, not a list
            orig_imgs = ops.convert_jittor2numpy_batch(orig_imgs)

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
