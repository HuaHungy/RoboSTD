"""
python scripts/main.py \
    --input /home/huahungy/hf_cache/lerobot/aloha_all_both_origin_50 \
    --output /home/huahungy/hf_cache/lerobot/aloha_all_both_origin_50/ \
    --rule agilex \
    --ops mirror
python scripts/main.py \
    --input /home/huahungy/hf_cache/lerobot/right_merged_50/ \
    --output /home/huahungy/hf_cache/lerobot/right_merged_50_mirrored/ \
    --rule agilex \
    --ops mirror

python scripts/main.py \
    --input /home/huaahungy/act/data/Agilex_Cobot_Magic_Put_the_bowl_on_the_plate_right_0509 \
    --output /home/huahungy/act/data/Agilex_Cobot_Magic_Put_the_bowl_on_the_plate_right_mirrored_0509 \
    --rule aloha \
    --ops mirror

python scripts/main.py \
    --input /home/huaahungy/act/data/Agilex_Cobot_Magic_Put_the_towel_in_the_basket_left_0511 \
    --output /home/huahungy/act/data/Agilex_Cobot_Magic_Put_the_towel_in_the_basket_left_mirrored_0511 \
    --rule aloha \
    --ops mirror

python scripts/main.py \
    --input /home/huaahungy/act/data/Agilex_Cobot_Magic_Put_the_towel_in_the_basket_right_0509 \
    --output /home/huahungy/act/data/Agilex_Cobot_Magic_Put_the_towel_in_the_basket_right_mirrored_0509 \
    --rule aloha \
    --ops mirror

"""


import argparse
import sys
import os
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline_core import Pipeline

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def interactive_mode():
    print("\n=== RoboSTD Unified Pipeline ===")
    
    # Input
    default_input = "data/raw"
    input_dir = input(f"Enter input dataset path (default: {default_input}): ").strip() or default_input
    
    # Output
    default_output = "data/output"
    output_dir = input(f"Enter output directory (default: {default_output}): ").strip() or default_output
    
    # Config
    default_config = "configs/default.yaml"
    config_path = input(f"Enter config path (default: {default_config}): ").strip() or default_config
    
    # Operations
    print("\nAvailable Operations:")
    print("1. Mirror (Left-Right Flip + Joint Mirror)")
    print("2. Reverse (Time Reverse)")
    print("3. Convert (Resolution/FPS)")
    print("4. Crop Half (Basic)")
    print("5. Crop Advanced (ROI)")
    print("6. Mask Mirror (Interactive)")
    # Stitch is omitted for single-stream pipeline simplicity
    
    ops_input = input("Enter operation numbers separated by comma (e.g. 1,2): ").strip()
    ops_map = {
        '1': 'mirror',
        '2': 'reverse',
        '3': 'convert',
        '4': 'crop_half',
        '5': 'crop_advanced',
        '6': 'mask_mirror'
    }
    
    operations = []
    for op_num in ops_input.split(','):
        op_num = op_num.strip()
        if op_num in ops_map:
            op_name = ops_map[op_num]
            params = {}
            
            # Ask for params based on op
            if op_name == 'convert':
                w = input("  [Convert] Target Width (optional): ").strip()
                h = input("  [Convert] Target Height (optional): ").strip()
                fps = input("  [Convert] Target FPS (optional): ").strip()
                if w: params['target_width'] = int(w)
                if h: params['target_height'] = int(h)
                if fps: params['target_fps'] = float(fps)
            
            elif op_name == 'crop_half':
                axis = input("  [Crop Half] Axis (vertical/horizontal, default: vertical): ").strip() or 'vertical'
                side = input("  [Crop Half] Keep Side (first/second, default: first): ").strip() or 'first'
                params['axis'] = axis
                params['keep_side'] = side
            
            elif op_name == 'crop_advanced':
                print("  [Crop Advanced] Interactive mode will be used for each video if no ROI provided.")
                # For batch processing, we usually want a fixed ROI.
                use_fixed = input("  Use fixed ROI for all videos? (y/n): ").strip().lower()
                if use_fixed == 'y':
                    roi_str = input("  Enter ROI (x,y,w,h): ").strip()
                    try:
                        params['roi'] = tuple(map(int, roi_str.split(',')))
                    except:
                        print("  Invalid ROI format. Using interactive.")
            
            operations.append({'name': op_name, 'params': params})
    
    # Rule
    rule = input("Enter robot rule name (e.g. aloha, realman): ").strip()
    
    print("\nStarting Pipeline...")
    pipeline = Pipeline(config_path, rule)
    pipeline.process_dataset(input_dir, output_dir, operations)

def main():
    parser = argparse.ArgumentParser(description="RoboSTD Unified Pipeline")
    parser.add_argument('--input', help="Input dataset directory")
    parser.add_argument('--output', help="Output directory")
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent.parent / "configs/default.yaml"), help="Path to config file")
    parser.add_argument('--ops', help="Comma separated operations (mirror,reverse,convert,crop_half,crop_advanced,mask_mirror,reconstruct)")
    parser.add_argument('--rule', help="Robot rule name (aloha, realman)")
    # Stage 2 (reconstruct) options
    parser.add_argument('--mirror-input', help="Mirrored dataset directory (required for reconstruct ops)")
    parser.add_argument('--source-arm', default=None, choices=['L', 'R'], help="Arm active in the single-arm demo (reconstruct)")
    parser.add_argument('--n-units', type=int, default=None, help="Number of manipulation units (reconstruct)")
    parser.add_argument('--language', default='', help="Task language instruction (reconstruct/LLM context)")
    parser.add_argument('--planner', default=None, choices=['default', 'openai'],
                        help="Coordination-constraint planner (reconstruct)")
    
    # If no args, interactive
    if len(sys.argv) == 1:
        interactive_mode()
        return

    args = parser.parse_args()
    
    if not args.input or not args.output:
        print("Error: --input and --output are required in non-interactive mode.")
        return
        
    operations = []
    if args.ops:
        for op in args.ops.split(','):
            operations.append({'name': op.strip()})

    # Stage 2: pseudo-bimanual reconstruction from orig + mirrored pair.
    if 'reconstruct' in [op['name'] for op in operations]:
        if not args.mirror_input:
            print("Error: --mirror-input is required for the reconstruct operation.")
            return
        pipeline = Pipeline(args.config, args.rule)
        pipeline.reconstruct_dataset(
            args.input, args.mirror_input, args.output,
            n_units=args.n_units, source_arm=args.source_arm, language=args.language,
            planner_name=args.planner,
        )
        return

    pipeline = Pipeline(args.config, args.rule)
    pipeline.process_dataset(args.input, args.output, operations)

if __name__ == "__main__":
    main()
