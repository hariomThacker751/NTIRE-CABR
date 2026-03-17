import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import json
from tqdm import tqdm
import lpips
import random
import numpy as np
import cv2

# Import base model and new HAFT utilities
from method.model import Bokehlicious
from method.nn_util import ApertureEncoder, FocalPriorGenerator, FusionStem
from method.blocks import ResidualBlock
from torch.nn import LayerNorm

# ---------------------------------------------------------
# 1. HAFT WRAPPER MODEL
# ---------------------------------------------------------
class HAFT_Bokehlicious(Bokehlicious):
    def __init__(self, d_embed=64, use_refinement=True, **kwargs):
        super().__init__(**kwargs)
        self.d_embed = d_embed
        self.use_refinement = use_refinement
        
        # Phase 1: Aperture Encoder
        self.aperture_encoder = ApertureEncoder(d_embed=self.d_embed)
        
        # Phase 2: Refinement Head
        if self.use_refinement:
            self.prior_gen = FocalPriorGenerator()
            self.fusion_stem = FusionStem(in_channels=7, d_embed=64)
            # Reusing your own ResidualBlock as the Refinement Transformer!
            self.refinement_blocks = ResidualBlock(
                embed_dim=64, embed_dim_next=64, depth=2, num_heads=2, 
                init_value=2, heads_range=6, ffn_dim=128, norm_layer=LayerNorm, use_pos_map=False
            )
            self.refinement_out = nn.Conv2d(64, self.in_chans, kernel_size=3, padding=1)
            # Zero-init so refinement starts as no-op (base_out + 0)
            nn.init.zeros_(self.refinement_out.weight)
            nn.init.zeros_(self.refinement_out.bias)

    def forward(self, source, bokeh_strength=None, pos_map=None, bokeh_strength_map=None, depth=None, mask=None, **kwargs):
        # 1. Generate Aperture Embedding (e_f)
        f_val = bokeh_strength.view(-1, 1) if bokeh_strength is not None else torch.ones(source.shape[0], 1).to(source.device)
        e_f = self.aperture_encoder(f_val)

        # --- Base CNN Forward Pass ---
        self.mean = self.mean.type_as(source)
        source_mean = (source - self.mean) * self.img_range

        x = torch.cat((source_mean, pos_map), dim=1) if self.in_stage_use_pos_map else source
        x = torch.cat((x, bokeh_strength_map), dim=1) if self.in_stage_use_bokeh_strength_map else x
        
        # Embed CoC map directly into the base CNN
        if getattr(self, 'use_coc_map', True) and kwargs.get('coc_map') is not None:
            x = torch.cat((x, kwargs['coc_map']), dim=1)

        x = self.in_stage(x)
        x = self.in_stage_2(x)

        encs = []
        for encoder, down, use_pos_map_e, depth_lvl in zip(self.encoders, self.downs, self.enc_blks_use_pos_map, range(0, self.u_depth)):
            pos_map_e = torch.nn.functional.interpolate(pos_map, scale_factor=1 / 2 ** depth_lvl, mode='bilinear') if pos_map is not None else None
            for blk in encoder:
                x = blk(x, pos_map=pos_map_e, e_f=e_f)
            encs.append(x)
            x = down(x)

        x_prep = self.conv_prep(x)
        pos_map_t = torch.nn.functional.interpolate(pos_map, scale_factor=1 / 2 ** self.u_depth, mode='bilinear') if self.positional_dfe else None
        
        # Deep Feature Extraction
        x_after_body = self.forward_features(x_prep, bokeh_strength=bokeh_strength, pos_map=pos_map_t)
        res = self.conv_after_body(x_after_body) + x_prep

        res = torch.cat((res, pos_map_t), dim=1) if self.positional_conv_last else res
        x = x + self.conv_last(res)

        for decoder, up, skip, enc_skip, use_pos_map_d, depth_lvl in zip(self.decoders, self.ups, self.skips, encs[::-1], self.dec_blks_use_pos_map, range(0, self.u_depth).__reversed__()):
            pos_map_d = torch.nn.functional.interpolate(pos_map, scale_factor=1 / 2 ** depth_lvl, mode='bilinear') if use_pos_map_d else None
            x = up(x)
            x = skip(x, enc_skip)
            for blk in decoder:
                x = blk(x, pos_map=pos_map_d, e_f=e_f)

        x = torch.cat((x, pos_map), dim=1) if self.out_stage_use_pos_map else x
        x = self.out_stage(x)
        base_out = (x / self.img_range + self.mean) + source
        base_out = base_out.clamp(0, 1)

        # --- HAFT Refinement Pass ---
        if self.use_refinement and depth is not None and mask is not None:
            F_map, W_focus = self.prior_gen(depth, mask)
            fused = self.fusion_stem(base_out, depth, mask, F_map, W_focus)
            
            fused_size = (fused.shape[2], fused.shape[3])
            fused_bhwc = fused.permute(0, 2, 3, 1)
            refined_features_bhwc = self.refinement_blocks(fused_bhwc, fused_size, pos_map=None, att_range_factor=bokeh_strength)
            refined_features = refined_features_bhwc.permute(0, 3, 1, 2)
            
            refinement_delta = self.refinement_out(refined_features)
            final_out = base_out + (refinement_delta * (1 - W_focus))
            return final_out
        
        return base_out


# ---------------------------------------------------------
# 2. CONFIGURATION (Small.pt backbone + 6GB VRAM safety)
# ---------------------------------------------------------
CONFIG = {
    "img_size": 512,           # Reduced from 512 → cuts compute ~44%, epochs ~4.5h instead of 8h
    "batch_size": 2,           # Reduced to 1 to fix OOM on 6GB GPU
    "accum_steps": 8,          # Effective batch size of 8 (less noisy gradients)
    "lr_backbone": 1e-5,       # Backbone: gentle fine-tune (already pretrained)
    "lr_haft": 5e-4,           # HAFT heads: learn aggressively (50x backbone)
    "epochs": 100,             # Total epochs
    "workers": 8,              # HPC system max
    "data_root": "dataset/RealBokeh_3MP", 
    
    "model_args": {
        "u_width": 32,                         
        "u_depth": 2,
        # --- Large backbone: 6 DFE blocks (not 4!) ---
        "embed_dims": [192, 192, 192, 192, 192, 192],            
        "depths": [6, 6, 6, 6, 6, 6],                   
        "num_heads": [6, 6, 6, 6, 6, 6],                
        "mlp_ratios": [2, 2, 2, 2, 2, 2],
        "init_values": [2, 2, 2, 2, 2, 2],
        "heads_ranges": [9, 9, 9, 9, 9, 9],
        "dec_blk_nums": [2, 4],
        "in_chans": 3,
        
        # --- Flags matching large backbone ---
        "in_stage_use_bokeh_strength_map": True,
        "positional_dfe": True,
        "positional_conv_last": True
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 3. REAL DATASET (Loading RGB, Depth, and Masks)
# ---------------------------------------------------------
class LocalBokehDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=256):
        self.root = os.path.join(root_dir, split)
        self.img_size = img_size
        self.split = split
        self.meta_dir = os.path.join(self.root, "metadata")
        self.to_tensor = transforms.ToTensor()
        
        self.samples = []
        if os.path.exists(self.meta_dir):
            json_files = sorted([f for f in os.listdir(self.meta_dir) if f.endswith('.json')])
            for jf in json_files:
                try:
                    with open(os.path.join(self.meta_dir, jf), 'r') as f:
                        meta = json.load(f)
                    src_path = os.path.join(self.root, meta['source_image'])
                    
                    for i, tgt_rel in enumerate(meta['target_images']):
                        tgt_path = os.path.join(self.root, tgt_rel)
                        self.samples.append({
                            "source": src_path,
                            "target": tgt_path,
                            "aperture": float(meta['target_avs'][i])
                        })
                except Exception:
                    pass

    def get_pos_map(self, w, h, i=0, j=0, crop_size=None):
        """
        Optimized coordinate map generation.
        If crop_size is provided, generates the 256x256 map directly.
        """
        if crop_size is not None:
            # Shift the coordinate range to match the crop
            # Map [0, w-1] to [0, 1]... but for the crop [j, j+crop_size-1]
            x_lin = torch.linspace(j/max(1, w-1), (j + crop_size - 1)/max(1, w-1), crop_size)
            y_lin = torch.linspace(1 - i/max(1, h-1), 1 - (i + crop_size - 1)/max(1, h-1), crop_size)
            return torch.meshgrid(x_lin, y_lin, indexing='xy')
        
        # Fallback for full size (validation)
        if w > h:
            crop_dist = (1 - (h / w)) / 2
            x_lin = torch.linspace(0, 1, w)
            y_lin = torch.linspace(1 - crop_dist, crop_dist, h)
            return torch.meshgrid(x_lin, y_lin, indexing='xy')
        else:
            x_lin = torch.linspace(0, 1, w)
            y_lin = torch.linspace(1, 0, h)
            return torch.meshgrid(x_lin, y_lin, indexing='xy')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Images
        sharp = Image.open(sample['source']).convert('RGB')
        bokeh = Image.open(sample['target']).convert('RGB')
        
        # 2. Derive Depth and Mask Paths
        base_name = os.path.splitext(os.path.basename(sample['source']))[0]
        depth_path = os.path.join(self.root, "depth", base_name + ".png")
        mask_path = os.path.join(self.root, "mask", base_name + ".png")
        
        # Load Depth as 16-bit and normalize, Mask as grayscale
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is not None and depth_raw.dtype == np.uint16:
            depth_np = depth_raw.astype(np.float32) / 65535.0
        else:
            depth_img = Image.open(depth_path).convert('L')
            depth_np = np.array(depth_img, dtype=np.float32) / 255.0
        mask_img = Image.open(mask_path).convert('L')

        aperture = sample['aperture']
        ap_embedding = 2.0 / (aperture if aperture > 0 else 1.0)

        # 3. Use original full-resolution images for cropping (vital for correct blur scale)
        w, h = sharp.size
        
        # Calculate CoC Map: CoC = |Depth - Focal_Depth| / Aperture
        focal_depth = np.median(depth_np)
        coc_np = np.abs(depth_np - focal_depth) / (aperture if aperture > 0 else 1.0)
        coc_np = np.clip(coc_np, 0.0, 1.0) 

        # 4. Convert to Tensors
        sharp_t = self.to_tensor(sharp)
        bokeh_t = self.to_tensor(bokeh)
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).float()  # (1, H, W)
        mask_t = self.to_tensor(mask_img)
        coc_t = torch.from_numpy(coc_np).unsqueeze(0).float()      # (1, H, W)
        bokeh_map = torch.full((1, h, w), ap_embedding, dtype=torch.float32)

        # 5. Cropping & Augmentation (Optimized Map Generation)
        if self.split == "train":
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(sharp_t, output_size=(self.img_size, self.img_size))
            sharp_t = TF.crop(sharp_t, i, j, h_crop, w_crop)
            bokeh_t = TF.crop(bokeh_t, i, j, h_crop, w_crop)
            bokeh_map = TF.crop(bokeh_map, i, j, h_crop, w_crop)
            depth_t = TF.crop(depth_t, i, j, h_crop, w_crop)
            mask_t = TF.crop(mask_t, i, j, h_crop, w_crop)
            coc_t = TF.crop(coc_t, i, j, h_crop, w_crop)

            # Generate pos_map directly for the crop
            px, py = self.get_pos_map(w, h, i, j, self.img_size)
            pos_map = torch.cat((px.unsqueeze(0), py.unsqueeze(0)), dim=0)

            # Random horizontal flip
            if random.random() > 0.5:
                sharp_t = TF.hflip(sharp_t)
                bokeh_t = TF.hflip(bokeh_t)
                pos_map = TF.hflip(pos_map)
                bokeh_map = TF.hflip(bokeh_map)
                depth_t = TF.hflip(depth_t)
                mask_t = TF.hflip(mask_t)
                coc_t = TF.hflip(coc_t)
                
            # Random vertical flip
            if random.random() > 0.5:
                sharp_t = TF.vflip(sharp_t)
                bokeh_t = TF.vflip(bokeh_t)
                pos_map = TF.vflip(pos_map)
                bokeh_map = TF.vflip(bokeh_map)
                depth_t = TF.vflip(depth_t)
                mask_t = TF.vflip(mask_t)
                coc_t = TF.vflip(coc_t)
        else:
            sharp_t = TF.center_crop(sharp_t, (self.img_size, self.img_size))
            bokeh_t = TF.center_crop(bokeh_t, (self.img_size, self.img_size))
            bokeh_map = TF.center_crop(bokeh_map, (self.img_size, self.img_size))
            depth_t = TF.center_crop(depth_t, (self.img_size, self.img_size))
            mask_t = TF.center_crop(mask_t, (self.img_size, self.img_size))
            coc_t = TF.center_crop(coc_t, (self.img_size, self.img_size))
            
            # Center crop pos_map logic
            i_mid, j_mid = (h - self.img_size) // 2, (w - self.img_size) // 2
            px, py = self.get_pos_map(w, h, i_mid, j_mid, self.img_size)
            pos_map = torch.cat((px.unsqueeze(0), py.unsqueeze(0)), dim=0)

        return {
            "input": sharp_t,
            "target": bokeh_t,
            "aperture": torch.tensor(ap_embedding, dtype=torch.float32), 
            "pos_map": pos_map,
            "bokeh_strength_map": bokeh_map,
            "depth": depth_t,
            "mask": mask_t,
            "coc_map": coc_t
        }
# ---------------------------------------------------------
# 4. MEMORY-OPTIMIZED TRAINING LOOP
# ---------------------------------------------------------
def calc_psnr(pred, target):
    """Calculate PSNR between prediction and target tensors."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

# ---------------------------------------------------------
# ADVANCED LOSS FUNCTIONS
# ---------------------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smoother than L1, better gradients near zero)."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class FFTLoss(nn.Module):
    """Frequency domain loss to preserve high-frequency details (edges, textures)."""
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return torch.mean(torch.abs(pred_fft - target_fft))

def train():
    print(f"✅ Hardware: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else "⚠️ CPU Mode")
    
    train_set = LocalBokehDataset(CONFIG['data_root'], split="train", img_size=CONFIG["img_size"])
    val_set = LocalBokehDataset(CONFIG['data_root'], split="validation", img_size=CONFIG["img_size"])
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True, 
                              pin_memory=True, num_workers=CONFIG["workers"], persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, 
                            pin_memory=True, num_workers=CONFIG["workers"], persistent_workers=True)
    
    print(f"🏗️ Initializing HAFT Model (Small Backbone)...")
    print(f"📊 Train: {len(train_set)} samples | Val: {len(val_set)} samples")
    model = HAFT_Bokehlicious(**CONFIG["model_args"], use_refinement=True).to(device)

    # ---------------------------------------------------------
    # STATE RECOVERY (Auto-detect latest checkpoint)
    # ---------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoints")
    backbone_weights = os.path.join(checkpoint_dir, "large.pt")
    
    start_epoch = 1
    
    # Auto-detect the latest haft_small_ep*.pth checkpoint
    import glob, re
    ep_checkpoints = glob.glob(os.path.join(checkpoint_dir, "haft_large_ep*.pth"))
    latest_epoch = 0
    latest_checkpoint = None
    for cp in ep_checkpoints:
        match = re.search(r'haft_large_ep(\d+)\.pth', os.path.basename(cp))
        if match:
            ep_num = int(match.group(1))
            if ep_num > latest_epoch:
                latest_epoch = ep_num
                latest_checkpoint = cp
    
    if latest_checkpoint is not None:
        print(f"📡 Found latest checkpoint: Epoch {latest_epoch} ({latest_checkpoint})")
        try:
            state_dict = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model.load_state_dict(filtered_dict, strict=False)
            start_epoch = latest_epoch + 1
            print(f"🚀 Resuming HAFT training from Epoch {start_epoch}")
        except Exception as e:
            print(f"❌ Error loading Epoch {latest_epoch} weights: {e}")
            raise e
    elif os.path.exists(backbone_weights):
        print(f"📡 Loading backbone from: {backbone_weights}")
        try:
            state_dict = torch.load(backbone_weights, map_location=device, weights_only=False)
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            # --- Fix in_stage weight mismatch (pretrained=6ch, new=7ch with CoC) ---
            for key in ['in_stage.weight', 'in_stage.bias']:
                if key in state_dict and key in model_dict and state_dict[key].shape != model_dict[key].shape:
                    old_w = state_dict[key]
                    new_w = model_dict[key].clone()
                    if key.endswith('.weight'):
                        n_old = old_w.shape[1]
                        new_w[:, :n_old, :, :] = old_w
                    else:
                        new_w[:old_w.shape[0]] = old_w
                    filtered_dict[key] = new_w
                    print(f"  🔧 Partial load for {key}: {old_w.shape} → {new_w.shape}")
            
            model.load_state_dict(filtered_dict, strict=False)
            start_epoch = 1
            print(f"✅ Successfully loaded backbone. Starting from Epoch 1.")
        except Exception as e:
            print(f"❌ Error loading backbone weights: {e}")
            raise e
    else:
        print(f"⚠️ Warning: No checkpoints found. Training from scratch.")

    # UNFREEZE EVERYTHING - PHASE 2 JOINT TRAINING
    for name, param in model.named_parameters():
        param.requires_grad = True  # Make sure everything is trainable
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🔓 Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")
            
    # DIFFERENTIAL LEARNING RATES: backbone gentle, HAFT heads aggressive
    haft_keywords = ['aperture_encoder', 'prior_gen', 'fusion_stem', 'refinement']
    backbone_params = [p for n, p in model.named_parameters() 
                       if p.requires_grad and not any(h in n for h in haft_keywords)]
    haft_params = [p for n, p in model.named_parameters() 
                   if p.requires_grad and any(h in n for h in haft_keywords)]
    
    print(f"📊 Backbone params: {sum(p.numel() for p in backbone_params):,} | HAFT params: {sum(p.numel() for p in haft_params):,}")
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': CONFIG['lr_backbone']},
        {'params': haft_params, 'lr': CONFIG['lr_haft']},
    ], weight_decay=1e-4)
    remaining_epochs = CONFIG["epochs"] - start_epoch + 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)
    
    # ---------------------------------------------------------
    # STOCHASTIC WEIGHT AVERAGING (SWA) SETUP
    # ---------------------------------------------------------
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-6)
    swa_start_epoch = int(CONFIG["epochs"] * 0.75) # Start SWA for the last 25% of epochs
    print(f"🔄 SWA will trigger starting from Epoch {swa_start_epoch}")
    
    char_loss = CharbonnierLoss(eps=1e-3).to(device)
    fft_loss = FFTLoss().to(device)
    lpips_loss = lpips.LPIPS(net="alex").to(device).eval()
    # bfloat16 does not need GradScaler (same exponent range as float32)
    best_psnr = 0.0
    best_lpips = float('inf')

    MAX_ITERS_PER_EPOCH = len(train_loader)
    print(f"🚀 Starting HAFT Training ({MAX_ITERS_PER_EPOCH} iters/epoch, LR_bb={CONFIG['lr_backbone']}, LR_haft={CONFIG['lr_haft']}, img_size={CONFIG['img_size']})...")

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        # ==================== TRAINING ====================
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{CONFIG['epochs']} [Train]", total=MAX_ITERS_PER_EPOCH)
        
        for i, batch in enumerate(pbar):
            if i >= MAX_ITERS_PER_EPOCH: break
            
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            ap = batch["aperture"].to(device, non_blocking=True)
            pos = batch["pos_map"].to(device, non_blocking=True)
            b_map = batch["bokeh_strength_map"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            coc = batch["coc_map"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(inp, bokeh_strength=ap, pos_map=pos, bokeh_strength_map=b_map, depth=depth, mask=mask, coc_map=coc)
                loss_char = char_loss(pred, tgt)

            # FFT + LPIPS are expensive - only compute on accumulation steps to save VRAM
            loss_extra = 0
            if (i + 1) % CONFIG["accum_steps"] == 0:
                pred_f32 = pred.float().clamp(0, 1)
                tgt_f32 = tgt.float()
                loss_extra = 0.1 * fft_loss(pred_f32, tgt_f32) + 0.3 * lpips_loss(pred_f32 * 2 - 1, tgt_f32 * 2 - 1).mean()

            loss = (loss_char + loss_extra) / CONFIG["accum_steps"]

            # NaN guard: skip corrupted batches instead of poisoning all gradients
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN/Inf loss at batch {i}, skipping...")
                optimizer.zero_grad()
                continue
            
            loss.backward()
            
            if (i + 1) % CONFIG["accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item() * CONFIG['accum_steps']
            epoch_loss += batch_loss
            pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "LR_bb": f"{optimizer.param_groups[0]['lr']:.1e}", "LR_haft": f"{optimizer.param_groups[1]['lr']:.1e}"})

        avg_train_loss = epoch_loss / len(train_loader)

        # ==================== VALIDATION ====================
        model.eval()
        val_psnr = 0.0
        val_loss = 0.0
        val_lpips_metric = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch}/{CONFIG['epochs']} [Val]"):
                inp = batch["input"].to(device, non_blocking=True)
                tgt = batch["target"].to(device, non_blocking=True)
                ap = batch["aperture"].to(device, non_blocking=True)
                pos = batch["pos_map"].to(device, non_blocking=True)
                b_map = batch["bokeh_strength_map"].to(device, non_blocking=True)
                depth = batch["depth"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                coc = batch["coc_map"].to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(inp, bokeh_strength=ap, pos_map=pos, bokeh_strength_map=b_map, depth=depth, mask=mask, coc_map=coc)
                    pred = pred.clamp(0, 1)
                    loss_v = char_loss(pred, tgt)
                
                # Calculate LPIPS (outside autocast for VGG stability)
                pred_f32 = pred.float()
                tgt_f32 = tgt.float()
                batch_lpips = lpips_loss(pred_f32 * 2 - 1, tgt_f32 * 2 - 1).mean().item()

                val_loss += loss_v.item()
                val_psnr += calc_psnr(pred, tgt)
                val_lpips_metric += batch_lpips
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_lpips = val_lpips_metric / len(val_loader)

        # ==================== SAVE ====================
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/haft_large_ep{epoch}.pth")
        
        # Save best models by metrics
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.state_dict(), "checkpoints/haft_large_best_psnr.pth")
            print(f"🏆 New best PSNR: {best_psnr:.2f} dB — saved haft_large_best_psnr.pth")
            
        if avg_val_lpips < best_lpips:
            best_lpips = avg_val_lpips
            torch.save(model.state_dict(), "checkpoints/haft_large_best_lpips.pth")
            print(f"🏆 New best LPIPS: {best_lpips:.4f} — saved haft_large_best_lpips.pth")
            
        # SWA Evaluation & Update
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            torch.save(swa_model.module.state_dict(), "checkpoints/haft_large_swa.pth")
            print(f"🔄 SWA Model updated and saved (haft_large_swa.pth)")
        else:
            scheduler.step()
            
        current_lr_bb = optimizer.param_groups[0]['lr']
        current_lr_haft = optimizer.param_groups[1]['lr']
        print(f"✅ Ep {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB | Val LPIPS: {avg_val_lpips:.4f} | LR_bb: {current_lr_bb:.1e} | LR_haft: {current_lr_haft:.1e}")

if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f"\n\n❌❌❌ TRAINING CRASHED ❌❌❌")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc()
        import sys
        sys.exit(1)