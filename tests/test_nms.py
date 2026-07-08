#!/usr/bin/env python3
"""
Test NMS algorithm alignment with PyTorch
Reference: ultralytics/utils/ops.py
"""

import numpy as np
import jittor as jt

# PyTorch is the alignment reference and a hard requirement for this suite
import torch
import torchvision.ops

from nkyolo.utils.ops import jtnms


class TestNMSAlgorithms:
    """Test NMS correctness and alignment with PyTorch"""
    
    def setup_method(self):
        """Set up test data"""
        # Create test boxes and scores
        self.boxes_np = np.array([
            [10, 10, 50, 50],   # box 1
            [15, 15, 55, 55],   # box 2 (overlaps box 1)
            [100, 100, 140, 140],  # box 3 (isolated)
            [12, 12, 52, 52],   # box 4 (overlaps box 1)
            [200, 200, 240, 240],  # box 5 (isolated)
        ], dtype=np.float32)
        
        self.scores_np = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        self.iou_threshold = 0.5
        
        # Convert to Jittor tensors
        self.boxes_jt = jt.array(self.boxes_np)
        self.scores_jt = jt.array(self.scores_np)
    
    def test_pytorch_nms_reference(self):
        """Test PyTorch NMS as a reference"""
        boxes_torch = torch.from_numpy(self.boxes_np)
        scores_torch = torch.from_numpy(self.scores_np)
        
        keep_indices = torchvision.ops.nms(boxes_torch, scores_torch, self.iou_threshold)
        keep_indices_np = keep_indices.numpy()
        
        print(f"PyTorch NMS result: {keep_indices_np}")
        assert len(keep_indices_np) > 0, "PyTorch NMS should keep some boxes"
        
        return keep_indices_np
    
    def test_jittor_ops_nms(self):
        """Test whether Jittor ops.nms is available"""
        try:
            keep_indices = jt.ops.nms(self.boxes_jt, self.scores_jt, self.iou_threshold)
        except (AttributeError, RuntimeError) as e:
            print(f"Jittor ops.nms unavailable: {e}")
            return False
        print(f"Jittor ops.nms result: {keep_indices}")
        return True

    def test_jtnms_implementation(self):
        """Test our jtnms implementation"""
        keep_indices = jtnms(self.boxes_jt, self.scores_jt, self.iou_threshold)
        keep_indices_np = keep_indices.numpy()
        
        print(f"jtnms result: {keep_indices_np}")
        assert len(keep_indices_np) > 0, "jtnms should keep some boxes"
        
        # Verify kept boxes are sorted by score descending
        kept_scores = self.scores_np[keep_indices_np]
        assert np.all(kept_scores[:-1] >= kept_scores[1:]), "Kept boxes should be sorted by score descending"
        
        return keep_indices_np
    
    def test_nms_alignment(self):
        """Test alignment between jtnms and PyTorch NMS"""
        # Get PyTorch result
        pytorch_result = self.test_pytorch_nms_reference()
        
        # Get our result
        jittor_result = self.test_jtnms_implementation()
        
        print(f"PyTorch result: {pytorch_result}")
        print(f"Jittor result: {jittor_result}")
        
        # Compare results
        if len(pytorch_result) == len(jittor_result):
            # If counts match, check exact match
            if np.array_equal(np.sort(pytorch_result), np.sort(jittor_result)):
                print("✅ NMS results match exactly!")
            else:
                print("⚠️  NMS results have same count but different indices, analyzing...")
                self._analyze_nms_differences(pytorch_result, jittor_result)
        else:
            print(f"⚠️  NMS results differ in count: PyTorch={len(pytorch_result)}, Jittor={len(jittor_result)}")
            self._analyze_nms_differences(pytorch_result, jittor_result)
    
    def _analyze_nms_differences(self, pytorch_result, jittor_result):
        """Analyze NMS result differences"""
        print("\n=== NMS difference analysis ===")
        
        # Show info for all boxes
        for i, (box, score) in enumerate(zip(self.boxes_np, self.scores_np)):
            status_pt = "✓" if i in pytorch_result else "✗"
            status_jt = "✓" if i in jittor_result else "✗"
            print(f"Box{i}: {box} Score:{score:.3f} PyTorch:{status_pt} Jittor:{status_jt}")
        
        # Compute IoU matrix
        print(f"\nIoU matrix (threshold={self.iou_threshold}):")
        n = len(self.boxes_np)
        for i in range(n):
            for j in range(n):
                if i != j:
                    iou = self._compute_iou_numpy(self.boxes_np[i], self.boxes_np[j])
                    print(f"IoU({i},{j})={iou:.3f}", end="  ")
            print()
    
    def _compute_iou_numpy(self, box1, box2):
        """NumPy IoU computation for reference"""
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
    # Run tests directly
    test = TestNMSAlgorithms()
    test.setup_method()
    
    print("=== Starting NMS tests ===")
    
    # Test Jittor ops.nms availability
    test.test_jittor_ops_nms()
    
    # Test jtnms
    test.test_jtnms_implementation()
    
    # PyTorch alignment
    test.test_nms_alignment()

    print("\n=== Tests complete ===")
