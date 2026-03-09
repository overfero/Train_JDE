# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path, PosixPath
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import DetMetrics, box_iou, ReIDMetrics
from ultralytics.utils.torch_utils import smart_inference_mode


class JDEValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a joint detection and embedding model.

    Example:
        ```python
        from ultralytics.models.yolo.jde import JDEValidator

        args = dict(model="yolov8n-jde.pt", data="coco8-seg.yaml")
        validator = JDEValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize JDEValidator and set task to 'jde', metrics to DetMetrics + ReIDMetrics."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "jde"
        self.metrics = DetMetrics()
        self.reid_metrics = ReIDMetrics()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Performs validation on the model and sets the epoch and best attributes."""
        if trainer is not None:
            self.epoch = trainer.epoch + 1
            self.best = trainer.best
        if model is not None:
            self.model_path = model if (isinstance(model, str) or isinstance(model, PosixPath)) else getattr(model, "pt_path", "")
        stats = super().__call__(trainer, model)
        return stats

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["tags"] = batch["tags"].to(self.device).float()
        return batch

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs.

        JDE.forward() returns (y, x) during inference where y is the decoded tensor.
        We extract y (index 0) and pass to NMS, then return list of dicts.
        """
        # JDE inference returns (decoded_tensor, raw_list) — take the decoded tensor
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        outputs = non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        # Return list of dicts compatible with DetectionValidator API
        # Each tensor: [x1, y1, x2, y2, conf, cls, embed...]
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        tags = batch["tags"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
            "tags": tags,
        }

    def _prepare_pred(self, pred, pbatch=None):
        """Prepares a batch of images and annotations for validation."""
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds, batch):
        """Metrics — updated for new DetectionValidator API (preds as list of dicts)."""
        batch_matched_tags = []  # List to store matched tags for each batch
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            tags = pbatch.pop("tags")
            cls = pbatch["cls"]
            bbox = pbatch["bboxes"]

            cls_pred = pred["cls"]
            conf_pred = pred["conf"]
            npr = len(cls_pred)
            nl = len(cls)

            matched_tags = torch.zeros(npr, dtype=torch.int, device=self.device)
            tp = np.zeros((npr, self.niou), dtype=bool)

            # Evaluate matches if both preds and GT exist
            if npr > 0 and nl > 0:
                tp, matched_tags = self._process_batch(pred, bbox, cls, tags)
                tp = tp.cpu().numpy()

            self.metrics.update_stats(
                {
                    "target_cls": cls.cpu().numpy(),
                    "target_img": cls.unique().cpu().numpy(),
                    "conf": np.zeros(0) if npr == 0 else conf_pred.cpu().numpy(),
                    "pred_cls": np.zeros(0) if npr == 0 else cls_pred.cpu().numpy(),
                    "tp": tp,
                }
            )

            if self.args.plots and npr > 0 and nl > 0:
                self.confusion_matrix.process_batch(pred, pbatch, conf=self.args.conf)

            batch_matched_tags.append(matched_tags)

            # Save
            if npr > 0 and (self.args.save_json or self.args.save_txt):
                predn_scaled = self.scale_preds(pred, pbatch)
                if self.args.save_json:
                    self.pred_to_json(predn_scaled, pbatch)
                if self.args.save_txt:
                    self.save_one_txt(
                        predn_scaled,
                        self.args.save_conf,
                        pbatch["ori_shape"],
                        self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                    )
        # Process batch for reid metrics
        reid_preds = [torch.cat((p["bboxes"], p["conf"][:, None], p["cls"][:, None], p["extra"]), dim=1) for p in preds]
        self.reid_metrics.process_batch(reid_preds, batch_matched_tags)


    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        reid_metrics = self.reid_metrics.get_metrics()
        detector_results = self.metrics.results_dict
        detector_results.update(reid_metrics)
        return detector_results

    def _process_batch(self, pred, gt_bboxes, gt_cls, gt_tags):
        """
        Return correct prediction matrix and matched tags.

        Args:
            pred (dict): Prediction dict with 'bboxes' and 'cls' keys.
            gt_bboxes (torch.Tensor): Ground-truth bounding boxes (M, 4).
            gt_cls (torch.Tensor): Ground-truth class indices (M,).
            gt_tags (torch.Tensor): Ground-truth tags (M,).

        Returns:
            (torch.Tensor): Correct prediction matrix (N, 10).
            (torch.Tensor): Matched tags tensor (N,).
        """
        iou = box_iou(gt_bboxes, pred["bboxes"])
        return self.match_predictions(pred["cls"], gt_cls, gt_tags, iou)

    def match_predictions(self, pred_classes, true_classes, true_tags, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            true_tags (torch.Tensor): Target tags of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values.
            use_scipy (bool): Whether to use scipy for matching.

        Returns:
            (torch.Tensor): Correct tensor of shape(N, 10) for 10 IoU thresholds.
            (torch.Tensor): Matched tags tensor of shape(N,).
        """
        matched_tags = [False] * pred_classes.shape[0]

        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()

        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                import scipy

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
                        if threshold == 0.5:
                            for gt_idx, pred_idx in zip(labels_idx[valid], detections_idx[valid]):
                                matched_tags[pred_idx] = true_tags[gt_idx].item()
            else:
                matches = np.nonzero(iou >= threshold)
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
                    if threshold == 0.5:
                        for gt_idx, pred_idx in matches:
                            matched_tags[pred_idx] = true_tags[gt_idx].item()

        return (
            torch.tensor(correct, dtype=torch.bool, device=pred_classes.device),
            torch.tensor(matched_tags, dtype=torch.int, device=pred_classes.device),
        )