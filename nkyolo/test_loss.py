import torch
import torch.nn as nn
import numpy as np
import importlib.util
import importlib
import sys
import os
import traceback

# ================= 0. 环境与冲突检测 =================
os.environ["JITTOR_LOG_LEVEL"] = "ERROR"

def check_naming_conflict():
    conflict_files = ['jittor_utils.py', 'jittor.py']
    cwd = os.getcwd()
    for f in conflict_files:
        if os.path.exists(os.path.join(cwd, f)):
            print(f"❌ 请重命名当前目录下的 {f}，否则会导致循环导入！")
            sys.exit(1)
check_naming_conflict()

try:
    import jittor as jt
    jt.flags.use_cuda = 0
except ImportError:
    print("❌ 未安装 Jittor: pip install jittor")
    sys.exit(1)

# ================= 1. 动态导入模块 =================
def load_modules():
    print(f"{'='*20} 加载模块 {'='*20}")
    
    # 1. 官方 PyTorch
    try:
        from ultralytics.utils import loss as official_loss_module
        print("✅ 官方 PyTorch Loss 导入成功")
    except ImportError:
        print("❌ 未安装 ultralytics")
        sys.exit(1)

    # 2. 本地 Jittor
    cwd = os.getcwd()
    local_utils_path = os.path.join(cwd, 'utils')
    if cwd not in sys.path: sys.path.insert(0, cwd)

    if not os.path.exists(local_utils_path):
         print(f"❌ 找不到 utils 目录: {local_utils_path}")
         sys.exit(1)
    
    init_file = os.path.join(local_utils_path, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f: f.write("")

    try:
        local_loss_module = importlib.import_module("utils.loss")
        importlib.reload(local_loss_module)
        print("✅ 本地 Jittor Loss 导入成功")
    except Exception as e:
        print(f"❌ 本地导入失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    return official_loss_module, local_loss_module

# ================= 2. 构造 Mock 模型 (维度精准对齐版) =================
class MockArgs:
    def __init__(self):
        self.box = 7.5
        self.cls = 0.5
        self.dfl = 1.5
        self.pose = 12.0
        self.kobj = 1.0
        self.label_smoothing = 0.0
        self.overlap_mask = True
        self.mask_ratio = 4

class MockHead(nn.Module):
    def __init__(self, nc=80, reg_max=16, task='detect'):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.stride = torch.tensor([8., 16., 32.])
        
        # --- 关键修改：根据任务计算通道数 no ---
        # Detect: box(4*reg) + cls
        self.no = nc + reg_max * 4
        
        # Segment 额外属性
        self.nm = 32 # number of masks
        if task == 'segment':
            self.no += self.nm
            
        # Pose 额外属性
        self.kpt_shape = [17, 3]
        if task == 'pose':
            self.no += (self.kpt_shape[0] * self.kpt_shape[1])

class MockHeadJittor(jt.Module):
    def __init__(self, nc=80, reg_max=16, task='detect'):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.stride = jt.array([8., 16., 32.])
        
        self.no = nc + reg_max * 4
        self.nm = 32
        if task == 'segment':
            self.no += self.nm
            
        self.kpt_shape = [17, 3]
        if task == 'pose':
            self.no += (self.kpt_shape[0] * self.kpt_shape[1])

class MockModelTorch(nn.Module):
    def __init__(self, nc=80, reg_max=16, task='detect'):
        super().__init__()
        self.args = MockArgs()
        self.head = MockHead(nc, reg_max, task)
        self.model = nn.ModuleList([self.head])
        self.nc = nc
        self.dummy_param = nn.Parameter(torch.zeros(1))

class MockModelJittor(jt.Module):
    def __init__(self, nc=80, reg_max=16, task='detect'):
        super().__init__()
        self.args = MockArgs()
        self.head = MockHeadJittor(nc, reg_max, task)
        self.model = [self.head]
        self.nc = nc

# ================= 3. 数据生成 (完全重构版) =================
def generate_data(batch_size=2, nc=80, reg_max=16, task='detect'):
    # 计算当前任务需要的通道数
    no = nc + reg_max * 4
    nm = 32
    nkpt = 17 * 3
    
    if task == 'segment': no += nm
    if task == 'pose': no += nkpt

    # PyTorch Data
    preds_pt = []
    # Jittor Data
    preds_jt = []
    
    if task == 'classify':
        p_np = np.random.randn(batch_size, nc).astype(np.float32)
        preds_pt = torch.from_numpy(p_np).requires_grad_(True)
        preds_jt = jt.array(p_np)
        
        batch_np = {
            'batch_idx': np.arange(batch_size).astype(np.float32),
            'cls': np.random.randint(0, nc, (batch_size,))
        }
    else:
        # Detect/Seg/Pose: 3 scales
        sizes = [80, 40, 20]
        preds_pt_list = []
        preds_jt_list = []
        
        for s in sizes:
            # [Batch, Channels, H, W]
            p_np = np.random.randn(batch_size, no, s, s).astype(np.float32)
            preds_pt_list.append(torch.from_numpy(p_np).requires_grad_(True))
            preds_jt_list.append(jt.array(p_np))
        
        # 对 Seg/Pose 任务，preds 结构可能不同
        # Detect: list[Tensor]
        # Segment: (list[Tensor], Tensor(proto)) 或 list[Tensor] 包含 proto?
        # 官方 v8SegmentationLoss 期望 preds 是 (feats, pred_masks, proto) 或者 feats
        # 这里为了简化，我们模拟 Detect 的最简输入，但如果 Loss 内部需要 tuple 结构，需要适配
        
        # 修正：Loss 内部通常自己 split 通道，所以只要 Tensor 通道数够就行
        # 但是 v8SegmentationLoss 的 __call__ 开头有：
        # feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        # 这意味着它期望输入是一个 tuple。我们需要构造这个 tuple。
        
        if task == 'segment':
            # 构造 Seg 需要的 tuple: (feats, pred_masks, proto)
            # 但 standard call 可能是 model(x)，这里我们只测 loss
            # Loss 内部代码：feats, pred_masks, proto = preds
            # 注意：官方 loss.py 逻辑比较杂，我们尽量模拟 training 时的输出
            
            # Proto masks: [B, 32, 160, 160]
            proto_np = np.random.randn(batch_size, nm, 160, 160).astype(np.float32)
            
            # 实际上 v8SegLoss 期望的是 tuple(feats, None, proto) 或者特定的结构
            # 让我们看代码：feats, pred_masks, proto = preds
            # feats 是 3 层特征图
            preds_pt = (preds_pt_list, None, torch.from_numpy(proto_np))
            preds_jt = (preds_jt_list, None, jt.array(proto_np))
            
        elif task == 'pose':
            # v8PoseLoss: feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
            # 只要传 list 就可以，kpts 在通道里 split 出来
            preds_pt = preds_pt_list
            preds_jt = preds_jt_list
        else:
            # Detect
            preds_pt = preds_pt_list
            preds_jt = preds_jt_list
            
        num_targets = 10
        batch_np = {
            'batch_idx': np.random.randint(0, batch_size, (num_targets,)).astype(np.float32),
            'cls': np.random.randint(0, nc, (num_targets, 1)).astype(np.float32),
            'bboxes': np.random.rand(num_targets, 4).astype(np.float32),
            'masks': np.random.rand(num_targets, 160, 160).astype(np.float32),
            'keypoints': np.random.rand(num_targets, 17, 3).astype(np.float32)
        }
        batch_np['bboxes'][:, 2:] *= 0.5 
        batch_np['bboxes'][:, :2] = batch_np['bboxes'][:, :2] * 0.5 + 0.25

    return preds_pt, preds_jt, batch_np

def to_torch_batch(batch_np, task='detect'):
    batch = {}
    for k, v in batch_np.items():
        if k == 'cls' and task == 'classify':
            batch[k] = torch.from_numpy(v).long()
        else:
            batch[k] = torch.from_numpy(v)
    return batch

def to_jittor_batch(batch_np):
    batch = {}
    for k, v in batch_np.items():
        batch[k] = jt.array(v)
    return batch

def smart_init(loss_cls, model_obj, framework_name):
    try:
        return loss_cls(model_obj)
    except TypeError as e:
        if "argument" in str(e) or "takes no arguments" in str(e):
            print(f"   ℹ️ {framework_name} 版 {loss_cls.__name__} 不接受 model 参数，尝试无参初始化...")
            try:
                inst = loss_cls()
                if hasattr(model_obj, 'nc'): inst.nc = model_obj.nc
                return inst
            except Exception as e2:
                raise RuntimeError(f"无参初始化也失败: {e2}")
        raise e

# ================= 4. 对比逻辑 =================
def compare_class(class_name, official_mod, local_mod):
    print(f"\n>>>>> 测试类: {class_name} <<<<<")
    if not hasattr(official_mod, class_name): return
    if not hasattr(local_mod, class_name): return

    # 1. 确定任务类型
    task = 'detect'
    if 'Segmentation' in class_name: task = 'segment'
    elif 'Pose' in class_name: task = 'pose'
    elif 'Classification' in class_name: task = 'classify'

    # 2. 初始化 (传入 task 以正确设置通道数)
    try:
        loss_cls_pt = getattr(official_mod, class_name)
        loss_cls_jt = getattr(local_mod, class_name)
        
        if class_name in ['BboxLoss', 'RotatedBboxLoss', 'DFLoss']:
            kwargs = {'reg_max': 16} if 'reg_max' in loss_cls_pt.__init__.__code__.co_varnames else {}
            loss_pt = loss_cls_pt(**kwargs)
            loss_jt = loss_cls_jt(**kwargs)
        elif class_name == 'KeypointLoss':
             loss_pt = loss_cls_pt(sigmas=np.ones(17))
             loss_jt = loss_cls_jt(sigmas=np.ones(17))
        else:
            loss_pt = smart_init(loss_cls_pt, MockModelTorch(task=task), "PyTorch")
            loss_jt = smart_init(loss_cls_jt, MockModelJittor(task=task), "Jittor")
            
    except Exception as e:
        print(f"❌ 初始化崩溃: {e}")
        # traceback.print_exc()
        return

    # 3. 准备数据
    if class_name in ['BboxLoss', 'DFLoss', 'FocalLoss', 'VarifocalLoss', 'KeypointLoss', 'RotatedBboxLoss']:
        print("ℹ️ 底层组件，跳过直接前向测试")
        return

    preds_pt, preds_jt, batch_np = generate_data(task=task)
    batch_pt = to_torch_batch(batch_np, task)
    batch_jt = to_jittor_batch(batch_np)

    # 4. 运行前向
    try:
        # PyTorch
        res_pt = loss_pt(preds_pt, batch_pt)
        # 【关键修复】处理返回值: (loss, items) -> loss.sum() -> item()
        if isinstance(res_pt, (tuple, list)):
            val_pt = res_pt[0]
        else:
            val_pt = res_pt
        
        # 如果是 0-d tensor，直接 item；如果是向量，先 sum
        if val_pt.numel() > 1:
            val_pt = val_pt.sum().item()
        else:
            val_pt = val_pt.item()

        # Jittor
        res_jt = loss_jt(preds_jt, batch_jt)
        if isinstance(res_jt, (tuple, list)):
            val_jt = res_jt[0]
        else:
            val_jt = res_jt
            
        if val_jt.numel() > 1:
            val_jt = val_jt.sum().item()
        else:
            val_jt = val_jt.item()

        diff = abs(val_pt - val_jt)
        print(f"Torch:  {val_pt:.6f}")
        print(f"Jittor: {val_jt:.6f}")
        
        if diff < 1e-2:
            print(f"Diff:   {diff:.6f} [✅ MATCH]")
        elif diff < 1.0:
             print(f"Diff:   {diff:.6f} [⚠️ CLOSE]")
        else:
             print(f"Diff:   {diff:.6f} [❌ DIFF]")

    except Exception as e:
        print(f"❌ 运行崩溃: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    t_mod, j_mod = load_modules()
    
    target_classes = [
        'v8DetectionLoss', 
        'v8SegmentationLoss', 
        'v8PoseLoss', 
        'v8ClassificationLoss', 
        'BboxLoss', 
        'DFLoss'
    ]
    
    for name in target_classes:
        compare_class(name, t_mod, j_mod)