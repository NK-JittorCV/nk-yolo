import jittor as jt
# import numpy as np

# Assume loss functions are in losses.py; adjust the import path as needed
from nkyolo.utils.loss import (
    VarifocalLoss, FocalLoss, DFLoss, BboxLoss, RotatedBboxLoss,
    KeypointLoss, v8DetectionLoss, v8SegmentationLoss,
    v8PoseLoss, v8ClassificationLoss, v8OBBLoss
)

# Enable Jittor GPU support (if available)
jt.flags.use_cuda = jt.has_cuda

class TestBaseLosses:
    """Test basic loss components"""
    
    def test_varifocal_loss(self):
        """Test Varifocal loss"""
        vfl = VarifocalLoss()
        pred_score = jt.array([[0.8, 0.2], [0.3, 0.7]], dtype=jt.float32)
        gt_score = jt.array([[1.0, 0.0], [0.0, 1.0]], dtype=jt.float32)
        label = jt.array([[1, 0], [0, 1]], dtype=jt.float32)
        
        loss = vfl.execute(pred_score, gt_score, label)
        
        assert loss.ndim == 0, "VarifocalLoss output should be scalar"
        assert loss > 0, "VarifocalLoss value should be > 0"
        print("VarifocalLoss test passed")

    def test_focal_loss(self):
        """Test Focal loss"""
        fl = FocalLoss()
        pred = jt.array([[0.8, 0.2], [0.3, 0.7]], dtype=jt.float32)
        label = jt.array([[1, 0], [0, 1]], dtype=jt.float32)
        
        loss = fl.execute(pred, label)
        
        assert loss.ndim == 0, "FocalLoss output should be scalar"
        assert loss > 0, "FocalLoss value should be > 0"
        print("FocalLoss test passed")

    def test_dfl_loss(self):
        """Test Distribution Focal loss"""
        reg_max = 16
        dfl = DFLoss(reg_max=reg_max)
        
        pred_dist = jt.randn(2, 4, reg_max)  # (num_anchors, 4, reg_max)
        target = jt.array([[5.5, 3.2, 7.8, 2.1], [1.2, 9.3, 4.5, 6.7]])
        
        loss = dfl(pred_dist, target)
        
        assert loss.shape == (2, 1), f"DFLoss shape mismatch, expected (2,1), got {loss.shape}"
        assert loss.sum() > 0, "DFLoss value should be > 0"
        print("DFLoss test passed")

    def test_bbox_loss(self):
        """Test bbox loss"""
        reg_max = 16
        bbox_loss = BboxLoss(reg_max=reg_max)
        
        batch_size = 1
        num_anchors = 10
        
        pred_dist = jt.randn(batch_size, num_anchors, reg_max*4)
        pred_bboxes = jt.randn(batch_size, num_anchors, 4)
        anchor_points = jt.randn(num_anchors, 2)
        target_bboxes = jt.array([[[0.2, 0.3, 0.7, 0.8]]])
        target_scores = jt.zeros(batch_size, num_anchors, 1)
        target_scores[0, 0, 0] = 1.0
        fg_mask = target_scores.sum(-1) > 0
        target_scores_sum = target_scores.sum()
        
        loss_iou, loss_dfl = bbox_loss(
            pred_dist, pred_bboxes, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        
        assert loss_iou > 0 and loss_dfl > 0, "Bbox loss should be > 0"
        print("BboxLoss test passed")

    def test_rotated_bbox_loss(self):
        """Test rotated bbox loss"""
        reg_max = 16
        rbbox_loss = RotatedBboxLoss(reg_max=reg_max)
        
        batch_size = 1
        num_anchors = 10
        
        pred_dist = jt.randn(batch_size, num_anchors, reg_max*4)
        pred_bboxes = jt.randn(batch_size, num_anchors, 5)  # Rotated boxes include an extra angle
        anchor_points = jt.randn(num_anchors, 2)
        target_bboxes = jt.array([[[0.2, 0.3, 0.5, 0.4, 0.785]]])  # Last value is angle (rad)
        target_scores = jt.zeros(batch_size, num_anchors, 1)
        target_scores[0, 0, 0] = 1.0
        fg_mask = target_scores.sum(-1) > 0
        target_scores_sum = target_scores.sum()
        
        loss_iou, loss_dfl = rbbox_loss(
            pred_dist, pred_bboxes, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        
        assert loss_iou > 0 and loss_dfl > 0, "Rotated bbox loss should be > 0"
        print("RotatedBboxLoss test passed")

    def test_keypoint_loss(self):
        """Test keypoint loss"""
        sigmas = jt.array([0.025, 0.025], dtype=jt.float32)
        kpt_loss = KeypointLoss(sigmas=sigmas)
        
        pred_kpts = jt.randn(2, 2, 3)  # (num_samples, num_kpts, 3)
        gt_kpts = jt.randn(2, 2, 3)
        kpt_mask = jt.array([[1, 1], [1, 0]], dtype=jt.float32)  # The second sample's second keypoint is invisible
        area = jt.array([[0.1], [0.2]], dtype=jt.float32)
        
        loss = kpt_loss(pred_kpts, gt_kpts, kpt_mask, area)
        
        assert loss.ndim == 0, "Keypoint loss should be scalar"
        assert loss > 0, "Keypoint loss should be > 0"
        print("KeypointLoss test passed")


class TestTaskLosses:
    """Test task-level loss functions"""
    
    def test_v8_detection_loss(self):
        """Test detection loss"""
        # Mock detection model
        class MockDetectionModel:
            def __init__(self):
                self.args = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
                self.model = [None, None, 
                            type('obj', (object,), {
                                'stride': jt.array([8, 16, 32]), 
                                'nc': 2, 
                                'reg_max': 16
                            })]
        
        model = MockDetectionModel()
        loss_fn = v8DetectionLoss(model)
        
        # Generate test data
        batch_size = 2
        feats = [
            jt.randn(batch_size, 2 + 16*4, 10, 10),  # (BS, no=2+64, H, W)
            jt.randn(batch_size, 2 + 16*4, 5, 5)
        ]
        
        batch = {
            "batch_idx": jt.array([0, 0, 1]),
            "cls": jt.array([0, 1, 0]),
            "bboxes": jt.array([[0.1, 0.2, 0.3, 0.4], 
                              [0.5, 0.6, 0.2, 0.3], 
                              [0.2, 0.3, 0.4, 0.5]])
        }
        
        total_loss, loss_items = loss_fn(feats, batch)
        
        assert total_loss.ndim == 0, "Detection total loss should be scalar"
        assert len(loss_items) == 3, "Detection loss should have 3 components"
        print("v8DetectionLoss test passed")

    def test_v8_classification_loss(self):
        """Test classification loss"""
        cls_loss = v8ClassificationLoss()
        
        pred = jt.randn(4, 3)  # 4 samples, 3 classes
        batch = {"cls": jt.array([0, 1, 2, 0])}  # Class labels
        
        loss, loss_items = cls_loss(pred, batch)
        
        assert loss.ndim == 0, "Classification total loss should be scalar"
        assert loss > 0, "Classification loss should be > 0"
        print("v8ClassificationLoss test passed")

    def test_v8_segmentation_loss(self):
        """Test segmentation loss"""
        # Mock segmentation model
        class MockSegmentModel:
            def __init__(self):
                self.args = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "overlap": False}
                self.model = [None, 
                            type('obj', (object,), {
                                'stride': jt.array([8, 16]), 
                                'nc': 2, 
                                'reg_max': 16
                            })]
        
        model = MockSegmentModel()
        loss_fn = v8SegmentationLoss(model)
        
        # Generate test data
        batch_size = 1
        feats = [jt.randn(batch_size, 2 + 16*4, 10, 10)]
        pred_masks = jt.randn(batch_size, 100, 32)  # (BS, anchors, 32)
        proto = jt.randn(batch_size, 32, 20, 20)    # Prototype masks
        preds = (feats, pred_masks, proto)
        
        batch = {
            "batch_idx": jt.array([0]),
            "cls": jt.array([0]),
            "bboxes": jt.array([[0.1, 0.2, 0.3, 0.4]]),
            "masks": jt.randint(0, 2, (1, 20, 20))  # GT mask
        }
        
        total_loss, loss_items = loss_fn(preds, batch)
        
        assert total_loss.ndim == 0, "Segmentation total loss should be scalar"
        assert len(loss_items) == 4, "Segmentation loss should have 4 components"
        print("v8SegmentationLoss test passed")

    def test_v8_obb_loss(self):
        """Test OBB detection loss"""
        # Mock OBB model
        class MockOBBModel:
            def __init__(self):
                self.args = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
                self.model = [None, 
                            type('obj', (object,), {
                                'stride': jt.array([8, 16]), 
                                'nc': 2, 
                                'reg_max': 16
                            })]
        
        model = MockOBBModel()
        loss_fn = v8OBBLoss(model)
        
        # Generate test data
        batch_size = 1
        feats = [jt.randn(batch_size, 2 + 16*4, 10, 10)]
        pred_angle = jt.randn(batch_size, 100, 1)  # Angle prediction
        preds = (feats, pred_angle)
        
        batch = {
            "batch_idx": jt.array([0]),
            "cls": jt.array([0]),
            "bboxes": jt.array([[0.1, 0.2, 0.3, 0.4, 0.785]])  # Last value is angle (rad)
        }
        
        total_loss, loss_items = loss_fn(preds, batch)
        
        assert total_loss.ndim == 0, "OBB total loss should be scalar"
        assert len(loss_items) == 3, "OBB loss should have 3 components"
        print("v8OBBLoss test passed")

    def test_v8_pose_loss(self):
        """Test pose loss"""
        # Mock pose model
        class MockPoseModel:
            def __init__(self):
                self.args = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 1.0, "kobj": 1.0}
                self.model = [None, 
                            type('obj', (object,), {
                                'stride': jt.array([8, 16]), 
                                'nc': 2, 
                                'reg_max': 16,
                                'kpt_shape': [2, 3]  # 2 keypoints, each has 3 params (x, y, visibility)
                            })]
        
        model = MockPoseModel()
        loss_fn = v8PoseLoss(model)
        
        # Generate test data
        batch_size = 1
        feats = [jt.randn(batch_size, 2 + 16*4, 10, 10)]
        pred_kpts = jt.randn(batch_size, 100, 2*3)  # Keypoint prediction
        preds = (feats, pred_kpts)
        
        batch = {
            "batch_idx": jt.array([0]),
            "cls": jt.array([0]),
            "bboxes": jt.array([[0.1, 0.2, 0.3, 0.4]]),
            "keypoints": jt.randn(1, 2, 3)  # (num_samples, num_kpts, 3)
        }
        
        total_loss, loss_items = loss_fn(preds, batch)
        
        assert total_loss.ndim == 0, "Pose total loss should be scalar"
        assert len(loss_items) == 5, "Pose loss should have 5 components"
        print("v8PoseLoss test passed")


if __name__ == "__main__":
    # Initialize test classes
    base_tester = TestBaseLosses()
    task_tester = TestTaskLosses()
    
    # Run basic loss tests
    print("=== Starting basic loss component tests ===")
    base_tester.test_varifocal_loss()
    base_tester.test_focal_loss()
    base_tester.test_dfl_loss()
    base_tester.test_bbox_loss()
    base_tester.test_rotated_bbox_loss()
    base_tester.test_keypoint_loss()
    
    # Run task-level loss tests
    print("\n=== Starting task-level loss function tests ===")
    task_tester.test_v8_detection_loss()
    task_tester.test_v8_classification_loss()
    task_tester.test_v8_segmentation_loss()
    task_tester.test_v8_obb_loss()
    task_tester.test_v8_pose_loss()
    
    print("\nAll tests passed!")
