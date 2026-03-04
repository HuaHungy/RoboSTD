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
    parser.add_argument('--config', default="configs/default.yaml", help="Path to config file")
    parser.add_argument('--ops', help="Comma separated operations (mirror,reverse,convert,crop_half,crop_advanced,mask_mirror)")
    parser.add_argument('--rule', help="Robot rule name (aloha, realman)")
    
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
            
    pipeline = Pipeline(args.config, args.rule)
    pipeline.process_dataset(args.input, args.output, operations)

if __name__ == "__main__":
    main()
