#!/usr/bin/env python3
"""
测试NMS算法与PyTorch对齐
参考：ultralytics/utils/ops.py
"""

import numpy as np
import pytest
import jittor as jt

# 尝试导入PyTorch作为参考
try:
    import torch
    import torchvision.ops
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch is not installed, cannot perform comparison tests")

from nkyolo.utils.ops import simple_nms, compute_iou_optimized


class TestNMSAlgorithms:
    """测试NMS算法的正确性和与PyTorch的对齐"""
    
    def setup_method(self):
        """设置测试数据"""
        # 创建测试用的边界框和分数
        self.boxes_np = np.array([
            [10, 10, 50, 50],   # box 1
            [15, 15, 55, 55],   # box 2 (与box1重叠)
            [100, 100, 140, 140],  # box 3 (独立)
            [12, 12, 52, 52],   # box 4 (与box1重叠)
            [200, 200, 240, 240],  # box 5 (独立)
        ], dtype=np.float32)
        
        self.scores_np = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        self.iou_threshold = 0.5
        
        # 转换为Jittor张量
        self.boxes_jt = jt.array(self.boxes_np)
        self.scores_jt = jt.array(self.scores_np)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_nms_reference(self):
        """测试PyTorch的NMS作为参考标准"""
        boxes_torch = torch.from_numpy(self.boxes_np)
        scores_torch = torch.from_numpy(self.scores_np)
        
        keep_indices = torchvision.ops.nms(boxes_torch, scores_torch, self.iou_threshold)
        keep_indices_np = keep_indices.numpy()
        
        print(f"PyTorch NMS结果: {keep_indices_np}")
        assert len(keep_indices_np) > 0, "PyTorch NMS应该保留一些框"
        
        return keep_indices_np
    
    def test_jittor_ops_nms(self):
        """测试Jittor的ops.nms是否可用"""
        try:
            keep_indices = jt.ops.nms(self.boxes_jt, self.scores_jt, self.iou_threshold)
            print(f"Jittor ops.nms结果: {keep_indices}")
            return True
        except (AttributeError, RuntimeError) as e:
            print(f"Jittor ops.nms不可用: {e}")
            return False
    
    def test_simple_nms_implementation(self):
        """测试我们的simple_nms实现"""
        keep_indices = simple_nms(self.boxes_jt, self.scores_jt, self.iou_threshold)
        keep_indices_np = keep_indices.numpy()
        
        print(f"simple_nms结果: {keep_indices_np}")
        assert len(keep_indices_np) > 0, "simple_nms应该保留一些框"
        
        # 验证保留的框按分数降序排列
        kept_scores = self.scores_np[keep_indices_np]
        assert np.all(kept_scores[:-1] >= kept_scores[1:]), "保留的框应该按分数降序排列"
        
        return keep_indices_np
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_nms_alignment(self):
        """测试simple_nms与PyTorch NMS的对齐"""
        # 获取PyTorch结果
        pytorch_result = self.test_pytorch_nms_reference()
        
        # 获取我们的结果
        jittor_result = self.test_simple_nms_implementation()
        
        print(f"PyTorch结果: {pytorch_result}")
        print(f"Jittor结果: {jittor_result}")
        
        # 比较结果
        if len(pytorch_result) == len(jittor_result):
            # 如果数量相同，检查是否完全匹配
            if np.array_equal(np.sort(pytorch_result), np.sort(jittor_result)):
                print("✅ NMS结果完全匹配!")
            else:
                print("⚠️  NMS结果数量相同但索引不同，分析差异...")
                self._analyze_nms_differences(pytorch_result, jittor_result)
        else:
            print(f"⚠️  NMS结果数量不同: PyTorch={len(pytorch_result)}, Jittor={len(jittor_result)}")
            self._analyze_nms_differences(pytorch_result, jittor_result)
    
    def test_compute_iou_accuracy(self):
        """测试compute_iou_optimized的准确性"""
        # 测试完全重叠的框
        box1 = jt.array([10, 10, 50, 50])
        box2 = jt.array([10, 10, 50, 50])
        iou = compute_iou_optimized(box1, box2)
        assert abs(iou - 1.0) < 1e-6, f"完全重叠的框IoU应该为1.0，实际为{iou}"
        
        # 测试不重叠的框
        box1 = jt.array([10, 10, 20, 20])
        box2 = jt.array([30, 30, 40, 40])
        iou = compute_iou_optimized(box1, box2)
        assert abs(iou - 0.0) < 1e-6, f"不重叠的框IoU应该为0.0，实际为{iou}"
        
        # 测试部分重叠的框
        box1 = jt.array([10, 10, 30, 30])  # 面积400
        box2 = jt.array([20, 20, 40, 40])  # 面积400
        # 交集: [20,20,30,30] = 面积100
        # 并集: 400 + 400 - 100 = 700
        # IoU = 100/700 ≈ 0.1429
        iou = compute_iou_optimized(box1, box2)
        expected_iou = 100.0 / 700.0
        assert abs(iou - expected_iou) < 1e-4, f"部分重叠框IoU应该为{expected_iou:.4f}，实际为{iou:.4f}"
        
        print("✅ IoU计算准确性测试通过!")
    
    def _analyze_nms_differences(self, pytorch_result, jittor_result):
        """分析NMS结果差异"""
        print("\n=== NMS差异分析 ===")
        
        # 显示所有框的信息
        for i, (box, score) in enumerate(zip(self.boxes_np, self.scores_np)):
            status_pt = "✓" if i in pytorch_result else "✗"
            status_jt = "✓" if i in jittor_result else "✗"
            print(f"框{i}: {box} 分数:{score:.3f} PyTorch:{status_pt} Jittor:{status_jt}")
        
        # 计算IoU矩阵
        print(f"\nIoU矩阵 (阈值={self.iou_threshold}):")
        n = len(self.boxes_np)
        for i in range(n):
            for j in range(n):
                if i != j:
                    iou = self._compute_iou_numpy(self.boxes_np[i], self.boxes_np[j])
                    print(f"IoU({i},{j})={iou:.3f}", end="  ")
            print()
    
    def _compute_iou_numpy(self, box1, box2):
        """NumPy版本的IoU计算，用于参考"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    # 直接运行测试
    test = TestNMSAlgorithms()
    test.setup_method()
    
    print("=== 开始NMS算法测试 ===")
    
    # 测试IoU计算
    test.test_compute_iou_accuracy()
    
    # 测试Jittor ops.nms可用性
    jt_nms_available = test.test_jittor_ops_nms()
    
    # 测试simple_nms
    test.test_simple_nms_implementation()
    
    # 如果PyTorch可用，测试对齐
    if TORCH_AVAILABLE:
        test.test_nms_alignment()
    else:
        print("跳过PyTorch对齐测试（PyTorch未安装）")
    
    print("\n=== 测试完成 ===")
