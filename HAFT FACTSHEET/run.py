# Package Imports
from os import makedirs
from statistics import mean
import torch
from torch import load, no_grad, clamp, Tensor
from torch.cuda import Event, synchronize, get_device_name
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from pathlib import Path
from shutil import make_archive, unpack_archive, copytree, rmtree
from datetime import datetime
from warnings import warn

# Local Imports
from dataset.loader import RealBokeh
from dataset.util import Mode

# ==========================================
# 🛑 TODO: IMPORT YOUR HAFT MODEL HERE 🛑
# Replace 'your_model_file' with the actual file where your HAFT class is defined.
from train_haft_small import HAFT_Bokehlicious as HAFT, CONFIG
# ==========================================

"""
!!!!! NTIRE CHALLENGE README: !!!!!
This script produces a submission ready .zip archive to be uploaded at 
https://www.codabench.org/competitions/12764/#/participate-tab for evaluation by our server.
"""

def unsqueeeze_batch(batch, device):
    for k, v in batch.items():
        if isinstance(v, Tensor):
            batch[k] = v.unsqueeze(0).to(device)
    return batch


# --- CONFIGURATION CLASS FOR EASY PATH PLACEMENT ---
class Config:
    phase = 'dev'                                             # 'dev' or 'test'
    dataset_root_dir = Path('./dataset')                      # Fixed: points to your root dataset folder
    out_path = Path('./outputs')                              # Path where outputs and zip will be saved
    name = 'HAFT_Large_Model_Dev'                               # Your model/architecture name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   # Auto-detect device
    image_format = 'png'                                      # 'png' is recommended
    extra_data = False                                        # True if you used extra training data
    checkpoint_path = Path("C:/Users/divya/Downloads/final-20260317T080406Z-1-001/HAFT FACTSHEET/checkpoints/haft_large_best_psnr.pth") # Path to your model weights
    max_inference_dim = 1024                                  # Max dimension for inference (saves VRAM & time)
# ---------------------------------------------------


if __name__ == "__main__":
    args = Config()
    checkpoint = args.checkpoint_path

    # Setup and sanity checking
    assert args.phase in ['dev', 'test'], f"Unknown argument for phase: {args.phase}, only ['dev', 'test'] are supported."
    assert checkpoint.is_file(), f"Checkpoint {checkpoint} is not a file."
    
    if args.image_format != 'png':
        warn(f"Image format {args.image_format} might lead to a lower score due to default compression behaviour, we recommend .png for final submission!")
             
    # FIXED DATASET PATH LOGIC
    # Maps directly to ./dataset/Bokeh_NTIRE2026 matching your VS Code structure
    dataset_path = args.dataset_root_dir / 'Bokeh_NTIRE2026' 
    assert args.dataset_root_dir.is_dir(), f"Path {args.dataset_root_dir} is not a directory."
    
    if args.phase == 'dev':
        if (dataset_path / 'validation').exists():
            print(f"Found NTIRE 2026 Bokeh Challenge Development inputs in: {dataset_path.absolute()}")
        else:
            print(f"Could not locate NTIRE 2026 Bokeh Challenge Development inputs ('validation' folder) in {dataset_path.absolute()}")
            err_msg = f"Could not find the 'validation' split at {dataset_path.absolute()}" if dataset_path.is_dir() else f"Could not find the 'Bokeh_NTIRE26' folder at {args.dataset_root_dir.absolute()}"
            assert (args.dataset_root_dir / 'Bokeh_NTIRE2026_Development_Inputs.zip').is_file(), f"{err_msg} OR the development inputs archive 'Bokeh_NTIRE2026_Development_Inputs.zip'."
            unpack_archive(args.dataset_root_dir / 'Bokeh_NTIRE2026_Development_Inputs.zip', args.dataset_root_dir)
            copytree(args.dataset_root_dir / 'Bokeh_NTIRE2026_Development_Inputs', args.dataset_root_dir, dirs_exist_ok=True)
            rmtree(args.dataset_root_dir / 'Bokeh_NTIRE2026_Development_Inputs')
            print("Successfully finished development dataset setup!")
    else:
        if (dataset_path / 'test').exists():
            print(f"Found NTIRE 2026 Bokeh Challenge Test inputs in: {dataset_path.absolute()}")
        else:
            print(f"Could not locate NTIRE 2026 Bokeh Challenge Test inputs ('test' folder) in {dataset_path.absolute()}")
            err_msg = f"Could not find the 'test' split at {dataset_path.absolute()}" if dataset_path.is_dir() else f"Could not find the 'Bokeh_NTIRE26' folder at {args.dataset_root_dir.absolute()}"
            assert (args.dataset_root_dir / 'Bokeh_NTIRE2026_Test_Inputs.zip').is_file(), f"{err_msg} OR the test inputs archive 'Bokeh_NTIRE2026_Test_Inputs.zip'."
            unpack_archive(args.dataset_root_dir / 'Bokeh_NTIRE2026_Test_Inputs.zip', args.dataset_root_dir)
            copytree(args.dataset_root_dir / 'Bokeh_NTIRE2026_Test_Inputs', args.dataset_root_dir, dirs_exist_ok=True)
            rmtree(args.dataset_root_dir / 'Bokeh_NTIRE2026_Test_Inputs')
            print("Successfully finished test dataset setup!")
            
    output_directory = args.out_path / args.name / 'NTIRE2026BokehChallenge' / args.phase
    try:
        rmtree(output_directory)
    except FileNotFoundError:
        pass
    makedirs(output_directory, exist_ok=False)
    print(f"Saving outputs to {output_directory.absolute()}")
    print(f"Running Architecture {args.name} on {'Development' if args.phase == 'dev' else 'Test'} set...")

    # ==========================================
    # 🛑 TODO: INITIALIZE YOUR HAFT MODEL HERE 🛑
    # Pass any necessary arguments your HAFT model requires
    model = HAFT(**CONFIG["model_args"], use_coc_map=True)
    # ==========================================

    print(f"Initialized model on {args.device}")

    # Load weights
    state_dict = load(checkpoint, weights_only=True) # Set weights_only=True to resolve the future warning
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    print(f"Loaded weights from {checkpoint.absolute()}")

    # Initialize evaluation dataset
    dataloader = RealBokeh(data_path=dataset_path, mode=Mode.VAL if args.phase=='dev' else Mode.TEST, device=args.device, challenge=True)
    try:
        if args.phase == 'dev':
            assert len(dataloader) == 78, f"There should be 78 images in the development set, but {len(dataloader)} were found."
    except AssertionError as error:
        warn(f"Incorrect input data found at {dataset_path / 'validation' if args.phase == 'dev' else 'test'}. Resetting directory!")
        rmtree(dataset_path / 'validation' if args.phase == 'dev' else 'test')
        raise error

    print(f"Initialized RealBokeh {'Development' if args.phase == 'dev' else 'Test'} phase dataloader")

    start_events = [Event(enable_timing=True) for _ in range(len(dataloader))]
    end_events = [Event(enable_timing=True) for _ in range(len(dataloader))]

    import numpy as np
    import cv2
    import torch
    from PIL import Image

    # Check once if depth directory exists
    split = 'validation' if args.phase=='dev' else 'test'
    depth_dir = dataset_path / split / 'depth'
    if not depth_dir.exists():
        warn(f"Depth directory not found at {depth_dir}. HAFT refinement will run on zero tensors (effectively disabled).")

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with no_grad():
            # --- Depth & CoC Map Integration ---
            image_name_str = batch['image_name']
            scene_id = image_name_str.split('_')[0]
            try:
                aperture = float(image_name_str.split('_f')[1])
            except:
                aperture = 2.0
                
            depth_path = depth_dir / f"{scene_id}_f22.png"
            _, h, w = batch['source'].shape
            
            if depth_path.exists():
                depth_img = Image.open(depth_path).convert('L')
                depth_np = np.array(depth_img, dtype=np.float32) / 255.0
                focal_depth = np.median(depth_np)
                
                # Calculate CoC Map
                coc_np = np.abs(depth_np - focal_depth) / (aperture if aperture > 0 else 1.0)
                coc_np = np.clip(coc_np, 0.0, 1.0)
                
                # Resize all maps to match source tensor shape
                depth_resized = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_AREA)
                coc_resized = cv2.resize(coc_np, (w, h), interpolation=cv2.INTER_AREA)
                
                batch['coc_map'] = torch.from_numpy(coc_resized).unsqueeze(0).float()
                batch['depth'] = torch.from_numpy(depth_resized).unsqueeze(0).float()
                
                # Synthesize mask from depth (no mask folder in validation set)
                mask_path = dataset_path / split / 'mask' / f"{scene_id}_f22.png"
                if mask_path.exists():
                    mask_img = Image.open(mask_path).convert('L')
                    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
                    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_AREA)
                    batch['mask'] = torch.from_numpy(mask_resized).unsqueeze(0).float()
                else:
                    mask_np = 1.0 - np.clip(np.abs(depth_resized - focal_depth) * 5.0, 0.0, 1.0)
                    batch['mask'] = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).float()
            else:
                batch['coc_map'] = torch.zeros((1, h, w), dtype=torch.float32)
                batch['depth'] = torch.zeros((1, h, w), dtype=torch.float32)
                batch['mask'] = torch.zeros((1, h, w), dtype=torch.float32)
            # -----------------------------------

            batch = unsqueeeze_batch(batch, args.device) 
            synchronize() 
            start_events[i].record() 
            
            # Build model kwargs (only pass keys the model expects)
            model_kwargs = {
                'source': batch['source'],
                'bokeh_strength': batch['bokeh_strength'],
                'pos_map': batch['pos_map'],
                'bokeh_strength_map': batch['bokeh_strength_map'],
                'depth': batch['depth'],
                'mask': batch['mask'],
                'coc_map': batch['coc_map'],
            }
            
            # --- Prediction Pass ---
            if args.device == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output_orig = model(**model_kwargs)
                
                # Horizontal flip pass
                model_kwargs_hflip = {}
                for k, v in model_kwargs.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 3:
                        model_kwargs_hflip[k] = v.flip(-1)
                    else:
                        model_kwargs_hflip[k] = v
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output_hflip = model(**model_kwargs_hflip)
                output_hflip = output_hflip.flip(-1)
            else:
                output_orig = model(**model_kwargs)
                
                model_kwargs_hflip = {}
                for k, v in model_kwargs.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 3:
                        model_kwargs_hflip[k] = v.flip(-1)
                    else:
                        model_kwargs_hflip[k] = v
                output_hflip = model(**model_kwargs_hflip).flip(-1)
            
            # Average predictions
            output = (output_orig + output_hflip) / 2.0
            
            end_events[i].record() 

            output = clamp(output, 0, 1)
            to_pil_image(output.squeeze(0).cpu()).save(output_directory / f"{batch['image_name']}.{args.image_format}")

    print("Finished prediction!")

    avg_time = mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)]) 
    print(f"Average time taken for {args.phase} phase: {avg_time / 1000:.3f} seconds on {get_device_name()}.")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {parameters / 1e6:.2f}M")

    metadata_file = output_directory / "readme.txt"
    with open(metadata_file, "w") as readme:
        readme.write(f"This file contains relevant metadata for the challenge leaderboard\n")
        readme.write(f"Architecture Name:{args.name}\n")
        readme.write(f"Parameters:{parameters / 1e6:.2f}M\n")
        readme.write(f"Runtime:{avg_time / 1000:.3f}s\n")
        readme.write(f"Device:{get_device_name()}\n")
        readme.write(f"Extra data:{'Yes' if args.extra_data else 'No'}\n")
        readme.write(f"Script Version:{1.2}\n")

    print(f"Wrote metadata to {metadata_file.absolute()}:")
    with open(metadata_file, "r") as readme:
        print(readme.read())

    print(f"Creating zip archive for Codabench submission...")
    archive_file = args.out_path / f'{args.name}_{args.phase}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    if args.phase == 'dev':
        image_names = [
            '8_f9.0.png', '6_f5.6.png', '23_f2.0.png', '21_f8.0.png', '17_f2.0.png', '2_f2.0.png', '25_f2.0.png',
            '28_f9.0.png', '6_f3.5.png', '9_f16.png', '27_f18.png', '28_f2.0.png', '5_f2.0.png', '15_f2.2.png',
            '24_f5.0.png', '18_f2.0.png', '5_f4.5.png', '28_f14.png', '21_f2.8.png', '21_f4.5.png', '6_f4.5.png',
            '19_f5.0.png', '29_f4.5.png', '24_f13.png', '21_f2.0.png', '14_f18.png', '2_f6.3.png', '28_f6.3.png',
            '18_f14.png', '13_f18.png', '10_f5.6.png', '10_f2.8.png', '14_f2.0.png', '12_f2.0.png', '22_f4.0.png',
            '17_f20.png', '24_f2.5.png', '24_f2.0.png', '29_f2.0.png', '7_f5.0.png', '26_f2.0.png', '2_f2.2.png',
            '25_f6.3.png', '3_f14.png', '2_f5.0.png', '27_f2.0.png', '10_f3.5.png', '16_f9.0.png', '1_f2.0.png',
            '20_f2.2.png', '30_f2.0.png', '22_f2.0.png', '5_f4.0.png', '15_f2.0.png', '30_f18.png', '8_f2.0.png',
            '19_f2.0.png', '27_f16.png', '3_f2.0.png', '7_f2.0.png', '4_f8.0.png', '9_f2.0.png', '1_f7.1.png',
            '12_f16.png', '20_f2.0.png', '27_f8.0.png', '10_f2.0.png', '13_f2.0.png', '11_f2.0.png', '11_f2.2.png',
            '26_f5.6.png', '13_f2.8.png', '13_f9.0.png', '5_f8.0.png', '6_f2.0.png', '16_f2.0.png', '4_f2.0.png',
            '23_f8.0.png'
        ]
        name_in_output = [file.name for file in output_directory.glob(f'*.{args.image_format}')]
        assert len(set(name_in_output) - set(image_names)) == 0, f"Unexpected images in output directory: {set(name_in_output) - set(image_names)}"
        assert len(set(image_names) - set(name_in_output)) == 0, f"Missing images from output directory: {set(image_names) - set(name_in_output)}"
        assert len(list(output_directory.glob(f'*.{args.image_format}'))) == 78

    assert metadata_file.exists(), "Could not find metadata file!"



    make_archive(archive_file, 'zip', root_dir=output_directory)
    print(f"Please upload your submission file found at {archive_file.absolute()}.zip to https://www.codabench.org/competitions/12764/#/participate-tab!")