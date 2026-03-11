# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn.functional as F
import numpy as np

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    # ── Pinned-memory buffer (dialokasi sekali, reused tiap frame) ────────────
    _pinned_buf: torch.Tensor | None = None
    _pinned_shape: tuple | None = None

    def preprocess(self, im):
        """GPU-native preprocess: pinned-memory upload → semua transform di CUDA.

        Mengganti pipeline CPU (np.stack → transpose → ascontiguousarray → from_numpy → .to(device))
        dengan pipeline yang jauh lebih cepat:
          1. np.stack sekali di CPU
          2. Transfer ke pinned memory (non-blocking DMA ke GPU)
          3. Semua transform (BGR→RGB, BHWC→BCHW, letterbox-pad, /255) di GPU
        """
        if isinstance(im, torch.Tensor):
            # Sudah tensor (dari LoadTensor), path lama
            t = im.to(self.device)
            return t.half() if self.model.fp16 else t.float()

        # ── 1. Stack numpy frames: list[(H,W,C)] → (N,H,W,C) uint8 ──────────
        arr = np.stack(im)  # (N,H,W,3) BGR uint8

        # ── 2. Upload ke GPU via pinned memory (async DMA) ────────────────────
        n, h, w, c = arr.shape
        if (
            DetectionPredictor._pinned_buf is None
            or DetectionPredictor._pinned_shape != arr.shape
        ):
            DetectionPredictor._pinned_buf = torch.empty(
                arr.shape, dtype=torch.uint8, pin_memory=True
            )
            DetectionPredictor._pinned_shape = arr.shape

        DetectionPredictor._pinned_buf.copy_(torch.from_numpy(arr))
        # non_blocking=True: DMA transfer terjadi paralel dengan CPU
        t = DetectionPredictor._pinned_buf.to(self.device, non_blocking=True)

        # ── 3. Semua transform di GPU ─────────────────────────────────────────
        # BGR → RGB  (flip channel axis, tidak ada memory copy baru)
        t = t.flip(3)  # (N,H,W,3)  BGR→RGB

        # NHWC → NCHW
        t = t.permute(0, 3, 1, 2).contiguous()  # (N,3,H,W)

        # Cast ke fp16 / fp32 + normalize
        t = t.half() if self.model.fp16 else t.float()
        t /= 255.0

        # ── 4. Letterbox resize+pad di GPU ────────────────────────────────────
        target_h, target_w = self.imgsz
        if (h, w) != (target_h, target_w):
            # Hitung scale dengan aspect-ratio preserving
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))

            # Bilinear resize di GPU
            t = F.interpolate(
                t, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

            # Pad ke target size (letterbox abu-abu 0.5)
            pad_top = (target_h - new_h) // 2
            pad_left = (target_w - new_w) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_right = target_w - new_w - pad_left
            # 114/255 ≈ 0.447 — nilai abu-abu standar letterbox YOLO
            t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), value=114.0 / 255.0)

        return t

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
