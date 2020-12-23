import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms_rotated, cat, nonzero_tuple
from detectron2.structures import Instances, Boxes, RotatedBoxes, pairwise_iou_rotated, ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..box_regression import Box2BoxTransformRotated
from ..poolers import ROIPooler
from ..matcher import SampleMatcher
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .rotated_fast_rcnn import RROIHeads

logger = logging.getLogger(__name__)


##########################################################################################################################################
############################################### Grasp Rotated RCNN Output ################################################################
##########################################################################################################################################


def grasp_fast_rcnn_inference_rotated(
    boxes, scores, tilts, zs, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    """
    Call `fast_rcnn_inference_single_image_rotated` for all images.
    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 5) if doing
            class-specific regression, or (Ri, 5) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        grasp_fast_rcnn_inference_single_image_rotated(
            scores_per_image, boxes_per_image, tilts_per_image, zs_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, tilts_per_image, zs_per_image, image_shape in zip(scores, boxes, tilts, zs, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def grasp_fast_rcnn_inference_single_image_rotated(
    scores, boxes, tilts, zs, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return rotated bounding-box detection results by thresholding
    on scores and applying rotated non-maximum suppression (Rotated NMS).
    Args:
        Same as `fast_rcnn_inference_rotated`, but with rotated boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference_rotated`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1) & torch.isfinite(tilts).all(dim=1) & torch.isfinite(zs).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        tilts = tilts[valid_mask]
        zs = zs[valid_mask]

    B = 5  # box dimension
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // B
    # Convert to Boxes to use the `clip` function ...
    boxes = RotatedBoxes(boxes.reshape(-1, B))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, B)  # R x C x B
    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    tilts = tilts[filter_inds[:, 0]]
    zs = zs[filter_inds[:, 0]]
    
    # Apply per-class Rotated NMS
    keep = batched_nms_rotated(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    tilts, zs = tilts[keep], zs[keep]
    
    result = Instances(image_shape)
    result.pred_boxes = RotatedBoxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.pred_zs = torch.flatten(zs)
    result.pred_tilts = torch.flatten(tilts)
    
    return result, filter_inds[:, 0]

     
class GraspRotatedFastRCNNOutputs(FastRCNNOutputs):
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        pred_zs,
        pred_tilts,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
    ):
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.pred_zs = torch.flatten(pred_zs)
        self.pred_tilts = torch.flatten(pred_tilts)
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        self.mse_loss = nn.MSELoss()
        
        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
                self.gt_zs = cat([p.gt_z for p in proposals], dim=0)
                self.gt_tilts = cat([p.gt_tilts for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found
    
    def z_mse_loss(self):
        return self.mse_loss(self.pred_zs, self.gt_zs)
        
    def tilt_mse_loss(self): 
        return self.mse_loss(self.pred_tilts, self.gt_tilts)
    
    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss(), "loss_mse_z": self.z_mse_loss(), "loss_mse_tilt": self.tilt_mse_loss()}


class GraspRotatedFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Rotated Fast R-CNN outputs.
    Edited for grasp planning.
    """

    @configurable
    def __init__(self, **kwargs):
        """
        NOTE: this interface is experimental.
        """
        super().__init__(**kwargs)
        self.z_pred = Linear(self.input_size, 1)
        self.tilt_pred = Linear(self.input_size, 1)
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        args = super().from_config(cfg, input_shape)
        args["box2box_transform"] = Box2BoxTransformRotated(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )
        args["loss_weight"] = {k: v for k, v in zip(["loss_cls", "loss_box_reg", "loss_mse_tilt", "loss_mse_z"], cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT)}
        return args

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        tilts = self.tilt_pred(x)
        zs = self.z_pred(x)
        return (scores, proposal_deltas), (tilts, zs)
    
    def losses(self, predictions, proposals, tilts_and_zs=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        tilts, zs = tilts_and_zs
        losses = GraspRotatedFastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            tilts, 
            zs,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    
    def inference(self, predictions, proposals, tilts_and_zs):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference_rotated`.
            list[Tensor]: same as `fast_rcnn_inference_rotated`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        tilts, zs = self.predict_tilts_zs(tilts_and_zs, proposals)
        image_shapes = [x.image_size for x in proposals]
        
        return grasp_fast_rcnn_inference_rotated(
            boxes,
            scores,
            tilts,
            zs,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )
        
    def predict_tilts_zs(self, tilts_and_zs, proposals):
        """
        Args:
            tilts_and_zs: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, 1), where Ri is the number of proposals for image i.
        """
        tilts, zs = tilts_and_zs
        num_inst_per_image = [len(p) for p in proposals]
        return tilts.split(num_inst_per_image), zs.split(num_inst_per_image)
        
        
@ROI_HEADS_REGISTRY.register()
class GraspRROIHeads(RROIHeads):
    """
    This class is used by Rotated Fast R-CNN to detect rotated boxes.
    For now, it only supports box predictions but not mask or keypoints.
    """

    @configurable
    def __init__(self, **kwargs):
        """
        NOTE: this interface is experimental.
        """
        super().__init__(**kwargs)
        assert (
            not self.mask_on and not self.keypoint_on
        ), "Mask/Keypoints not supported in Rotated ROIHeads."
        assert not self.train_on_pred_boxes, "train_on_pred_boxes not implemented for RROIHeads!"

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        ret["proposal_matcher"] = SampleMatcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=True,
        )
        return ret
        
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        assert pooler_type in ["ROIAlignRotated"], pooler_type
        # assume all channel counts are equal
        in_channels = [input_shape[f].channels for f in in_features][0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # This line is the only difference v.s. StandardROIHeads
        box_predictor = GraspRotatedFastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the RROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`StandardROIHeads.forward`

        Returns:
            list[Instances]: length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the rotated proposal boxes
                - gt_boxes: the ground-truth rotated boxes that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                - gt_classes: the ground-truth classification lable for each proposal
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou_rotated(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )

            match_quality_matrix = F.relu(match_quality_matrix)

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[sampled_targets]
                proposals_per_image.gt_z = targets_per_image.gt_z[sampled_targets]
                proposals_per_image.gt_tilts = targets_per_image.gt_tilts[sampled_targets]
            else:
                gt_boxes = RotatedBoxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 5))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_grasp(features, proposals)
            # losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_grasp(features, proposals)
            # pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_grasp(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances],
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the grasp prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions, tilts_and_zs = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, tilts_and_zs)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals, tilts_and_zs)
            return pred_instances