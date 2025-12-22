import jittor as jt
# import numpy as np

# 假设损失函数代码在losses.py文件中，根据实际情况修改导入路径
from nkyolo.utils.loss import (
    VarifocalLoss, FocalLoss, DFLoss, BboxLoss, RotatedBboxLoss,
    KeypointLoss, v8DetectionLoss, v8SegmentationLoss,
    v8PoseLoss, v8ClassificationLoss, v8OBBLoss
)

import ultralytics.utils.loss as loss

# 启用Jittor的GPU支持（如果可用）
jt.flags.use_cuda = jt.has_cuda

class TestBaseLosses:
    """测试基础损失组件"""
    
    def test_varifocal_loss(self):
        """测试变焦点损失"""
        vfl = VarifocalLoss()
        pred_score = jt.array([[0.8, 0.2], [0.3, 0.7]], dtype=jt.float32)
        gt_score = jt.array([[1.0, 0.0], [0.0, 1.0]], dtype=jt.float32)
        label = jt.array([[1, 0], [0, 1]], dtype=jt.float32)
        
        loss = vfl.execute(pred_score, gt_score, label)
        
        assert loss.ndim == 0, "VarifocalLoss输出应为标量"
        assert loss > 0, "VarifocalLoss值应大于0"
        print("VarifocalLoss测试通过")

    def test_focal_loss(self):
        """测试焦点损失"""
        fl = FocalLoss()
        pred = jt.array([[0.8, 0.2], [0.3, 0.7]], dtype=jt.float32)
        label = jt.array([[1, 0], [0, 1]], dtype=jt.float32)
        
        loss = fl.execute(pred, label)
        
        assert loss.ndim == 0, "FocalLoss输出应为标量"
        assert loss > 0, "FocalLoss值应大于0"
        print("FocalLoss测试通过")

    def test_dfl_loss(self):
        """测试分布焦点损失"""
        reg_max = 16
        dfl = DFLoss(reg_max=reg_max)
        
        pred_dist = jt.randn(2, 4, reg_max)  # (num_anchors, 4, reg_max)
        target = jt.array([[5.5, 3.2, 7.8, 2.1], [1.2, 9.3, 4.5, 6.7]])
        
        loss = dfl(pred_dist, target)
        
        assert loss.shape == (2, 1), f"DFLoss形状错误，预期(2,1)，实际{loss.shape}"
        assert loss.sum() > 0, "DFLoss值应大于0"
        print("DFLoss测试通过")

    def test_bbox_loss(self):
        """测试边界框损失"""
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
        
        assert loss_iou > 0 and loss_dfl > 0, "边界框损失应大于0"
        print("BboxLoss测试通过")

    def test_rotated_bbox_loss(self):
        """测试旋转边界框损失"""
        reg_max = 16
        rbbox_loss = RotatedBboxLoss(reg_max=reg_max)
        
        batch_size = 1
        num_anchors = 10
        
        pred_dist = jt.randn(batch_size, num_anchors, reg_max*4)
        pred_bboxes = jt.randn(batch_size, num_anchors, 5)  # 旋转框多一个角度
        anchor_points = jt.randn(num_anchors, 2)
        target_bboxes = jt.array([[[0.2, 0.3, 0.5, 0.4, 0.785]]])  # 最后一个是角度(rad)
        target_scores = jt.zeros(batch_size, num_anchors, 1)
        target_scores[0, 0, 0] = 1.0
        fg_mask = target_scores.sum(-1) > 0
        target_scores_sum = target_scores.sum()
        
        loss_iou, loss_dfl = rbbox_loss(
            pred_dist, pred_bboxes, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        
        assert loss_iou > 0 and loss_dfl > 0, "旋转边界框损失应大于0"
        print("RotatedBboxLoss测试通过")

    def test_keypoint_loss(self):
        """测试关键点损失"""
        sigmas = jt.array([0.025, 0.025], dtype=jt.float32)
        kpt_loss = KeypointLoss(sigmas=sigmas)
        
        pred_kpts = jt.randn(2, 2, 3)  # (num_samples, num_kpts, 3)
        gt_kpts = jt.randn(2, 2, 3)
        kpt_mask = jt.array([[1, 1], [1, 0]], dtype=jt.float32)  # 第二个样本的第二个关键点不可见
        area = jt.array([[0.1], [0.2]], dtype=jt.float32)
        
        loss = kpt_loss(pred_kpts, gt_kpts, kpt_mask, area)
        
        assert loss.ndim == 0, "关键点损失应为标量"
        assert loss > 0, "关键点损失应大于0"
        print("KeypointLoss测试通过")


class TestTaskLosses:
    """测试任务级损失函数"""
    
    def test_v8_detection_loss(self):
        """测试检测损失"""
        # 模拟检测模型
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
        
        # 生成测试数据
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
        
        assert total_loss.ndim == 0, "检测总损失应为标量"
        assert len(loss_items) == 3, "检测损失应包含3个分量"
        print("v8DetectionLoss测试通过")

    def test_v8_classification_loss(self):
        """测试分类损失"""
        cls_loss = v8ClassificationLoss()
        
        pred = jt.randn(4, 3)  # 4个样本，3个类别
        batch = {"cls": jt.array([0, 1, 2, 0])}  # 类别标签
        
        loss, loss_items = cls_loss(pred, batch)
        
        assert loss.ndim == 0, "分类总损失应为标量"
        assert loss > 0, "分类损失应大于0"
        print("v8ClassificationLoss测试通过")

    def test_v8_segmentation_loss(self):
        """测试分割损失"""
        # 模拟分割模型
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
        
        # 生成测试数据
        batch_size = 1
        feats = [jt.randn(batch_size, 2 + 16*4, 10, 10)]
        pred_masks = jt.randn(batch_size, 100, 32)  # (BS, anchors, 32)
        proto = jt.randn(batch_size, 32, 20, 20)    # 原型掩码
        preds = (feats, pred_masks, proto)
        
        batch = {
            "batch_idx": jt.array([0]),
            "cls": jt.array([0]),
            "bboxes": jt.array([[0.1, 0.2, 0.3, 0.4]]),
            "masks": jt.randint(0, 2, (1, 20, 20))  # GT掩码
        }
        
        total_loss, loss_items = loss_fn(preds, batch)
        
        assert total_loss.ndim == 0, "分割总损失应为标量"
        assert len(loss_items) == 4, "分割损失应包含4个分量"
        print("v8SegmentationLoss测试通过")

    def test_v8_obb_loss(self):
        """测试旋转目标检测损失"""
        # 模拟OBB模型
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
        
        # 生成测试数据
        batch_size = 1
        feats = [jt.randn(batch_size, 2 + 16*4, 10, 10)]
        pred_angle = jt.randn(batch_size, 100, 1)  # 角度预测
        preds = (feats, pred_angle)
        
        batch = {
            "batch_idx": jt.array([0]),
            "cls": jt.array([0]),
            "bboxes": jt.array([[0.1, 0.2, 0.3, 0.4, 0.785]])  # 最后一个是角度(rad)
        }
        
        total_loss, loss_items = loss_fn(preds, batch)
        
        assert total_loss.ndim == 0, "OBB总损失应为标量"
        assert len(loss_items) == 3, "OBB损失应包含3个分量"
        print("v8OBBLoss测试通过")

    def test_v8_pose_loss(self):
        """测试姿态估计损失"""
        # 模拟姿态模型
        class MockPoseModel:
            def __init__(self):
                self.args = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 1.0, "kobj": 1.0}
                self.model = [None, 
                            type('obj', (object,), {
                                'stride': jt.array([8, 16]), 
                                'nc': 2, 
                                'reg_max': 16,
                                'kpt_shape': [2, 3]  # 2个关键点，每个有3个参数(x,y,可见性)
                            })]
        
        model = MockPoseModel()
        loss_fn = v8PoseLoss(model)
        
        # 生成测试数据
        batch_size = 1
        feats = [jt.randn(batch_size, 2 + 16*4, 10, 10)]
        pred_kpts = jt.randn(batch_size, 100, 2*3)  # 关键点预测
        preds = (feats, pred_kpts)
        
        batch = {
            "batch_idx": jt.array([0]),
            "cls": jt.array([0]),
            "bboxes": jt.array([[0.1, 0.2, 0.3, 0.4]]),
            "keypoints": jt.randn(1, 2, 3)  # (num_samples, num_kpts, 3)
        }
        
        total_loss, loss_items = loss_fn(preds, batch)
        
        assert total_loss.ndim == 0, "姿态总损失应为标量"
        assert len(loss_items) == 5, "姿态损失应包含5个分量"
        print("v8PoseLoss测试通过")


if __name__ == "__main__":
    # 初始化测试类
    base_tester = TestBaseLosses()
    task_tester = TestTaskLosses()
    
    # 运行基础损失测试
    print("=== 开始基础损失组件测试 ===")
    base_tester.test_varifocal_loss()
    base_tester.test_focal_loss()
    base_tester.test_dfl_loss()
    base_tester.test_bbox_loss()
    base_tester.test_rotated_bbox_loss()
    base_tester.test_keypoint_loss()
    
    # 运行任务级损失测试
    print("\n=== 开始任务级损失函数测试 ===")
    task_tester.test_v8_detection_loss()
    task_tester.test_v8_classification_loss()
    task_tester.test_v8_segmentation_loss()
    task_tester.test_v8_obb_loss()
    task_tester.test_v8_pose_loss()
    
    print("\n所有测试通过！")
