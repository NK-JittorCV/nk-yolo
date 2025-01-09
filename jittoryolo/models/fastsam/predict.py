import jittor as jt
from PIL import Image

from jittoryolo.models.yolo.segment import SegmentationPredictor
from jittoryolo.utils import DEFAULT_CFG, checks
from jittoryolo.utils.metrics import box_iou
from jittoryolo.utils.ops import scale_masks

from .utils import adjust_bboxes_to_image_border


class FastSAMPredictor(SegmentationPredictor):
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks in jittoryolo
    YOLO framework.

    This class extends the SegmentationPredictor, customizing the prediction pipeline specifically for fast SAM. It
    adjusts post-processing steps to incorporate mask prediction and non-max suppression while optimizing for single-
    class segmentation.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes a FastSAMPredictor for fast SAM segmentation tasks in jittoryolo YOLO framework."""
        super().__init__(cfg, overrides, _callbacks)
        self.prompts = {}

    def postprocess(self, preds, img, orig_imgs):
        """Applies box postprocess for FastSAM predictions."""
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = jt.array([0, 0, result.orig_shape[1], result.orig_shape[0]], dtype=jt.float32).to(preds[0].device)
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            iou = box_iou(full_box.unsqueeze(0), boxes)
            idx = jt.nonzero(iou > 0.9).flatten()
            if idx.size > 0:
                mask = jt.zeros_like(boxes[:, 0], dtype=jt.bool)
                mask = mask.index_set(idx, jt.ones_like(idx, dtype=jt.bool))
                updated_boxes = boxes.clone()
                updated_boxes[mask] = full_box
                result.boxes.xyxy = updated_boxes

        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """
        Internal function for image segmentation inference based on cues like bounding boxes, points, and masks.
        Leverages SAM's specialized architecture for prompt-based, real-time segmentation.

        Args:
            results (Results | List[Results]): The original inference results from FastSAM models without any prompts.
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.
            texts (str | List[str], optional): Textual prompts, a list contains string objects.

        Returns:
            (List[Results]): The output results determined by prompts.
        """
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results = []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if len(result) == 0:
                prompt_results.append(result)
                continue
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = scale_masks(masks[None], result.orig_shape)[0]
            # bboxes prompt
            idx = jt.zeros(len(result), dtype=jt.bool).to(self.device)
            if bboxes is not None:
                bboxes = jt.array(bboxes, dtype=jt.int32).to(self.device)
                bboxes = jt.unsqueeze(bboxes, 0) if bboxes.ndim == 1 else bboxes
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                mask_areas = jt.stack([masks[:, b[1]:b[3], b[0]:b[2]].sum(dim=(1, 2)) for b in bboxes])
                full_mask_areas = masks.sum(dim=(1, 2))
                union = bbox_areas.unsqueeze(1) + full_mask_areas - mask_areas
                #TODO：index_set有问题
                idx = idx.index_set(jt.argmax(mask_areas / union, dim=1), jt.ones_like(jt.argmax(mask_areas / union, dim=1), dtype=jt.bool))
               
            if points is not None:
                points = jt.array(points, dtype=jt.int32).to(self.device)
                points = jt.unsqueeze(points, 0) if points.ndim == 1 else points
                if labels is None:
                    labels = jt.ones(points.shape[0])
                labels = jt.array(labels, dtype=jt.int32).to(self.device)
                assert len(labels) == len(points), f"Expected `labels` to have the same size as `points`, but got {len(labels)} and {len(points)}"
                point_idx = jt.ones(len(result), dtype=jt.bool).to(self.device) if labels.sum() == 0 else jt.zeros(len(result), dtype=jt.bool).to(self.device)
                for point, label in zip(points, labels):
                    indices = jt.nonzero(masks[:, point[1], point[0]])
                    #TODO：index_set有问题
                    point_idx = point_idx.index_set(indices, jt.ones_like(indices, dtype=jt.bool) * bool(label))
                idx = idx | point_idx

            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if masks[i].sum() <= 100:
                        filter_idx.append(i)
                        continue
                    crop_ims.append(Image.fromarray(result.orig_img[y1:y2, x1:x2, ::-1]))
                similarity = self._clip_inference(crop_ims, texts)
                text_idx = jt.argmax(similarity, dim=-1)
                if len(filter_idx):
                    #TODO:int(text_idx) 这一部分在 Jittor 中可能需要根据 text_idx 的具体形状和类型进行调整，确保其能够正确转换为整数。如果 text_idx 是一个张量，请考虑使用适当的方法提取其值。
                    text_idx += (jt.array(filter_idx, dtype=jt.int32).to(self.device)[None] <= int(text_idx)).sum(0)
                #TODO：index_set有问题
                idx = idx.index_set(text_idx, jt.ones_like(text_idx, dtype=jt.bool))


            prompt_results.append(result[idx])

        return prompt_results

    def _clip_inference(self, images, texts):
        """
        CLIP Inference process.

        Args:
            images (List[PIL.Image]): A list of source images and each of them should be PIL.Image type with RGB channel order.
            texts (List[str]): A list of prompt texts and each of them should be string object.

        Returns:
            (torch.Tensor): The similarity between given images and texts.
        """
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/jittoryolo/CLIP.git")
            import clip
        if (not hasattr(self, "clip_model")) or (not hasattr(self, "clip_preprocess")):
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        #TODO:请确认 self.clip_preprocess(image) 返回的是 Jittor 张量，并且 to(self.device) 在 Jittor 中正确移动张量到指定设备。
        images = jt.stack([self.clip_preprocess(image).to(self.device) for image in images])
        tokenized_text = clip.tokenize(texts).to(self.device)
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # (N, 512)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # (M, 512)
        return (image_features * text_features[:, None]).sum(-1)  # (M, N)

    def set_prompts(self, prompts):
        """Set prompts in advance."""
        self.prompts = prompts
