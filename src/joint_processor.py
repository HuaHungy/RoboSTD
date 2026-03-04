import os
import yaml
import json
import shutil
import pandas as pd
import numpy as np

class JointProcessor:
    def __init__(self, config_path=None):
        self.config = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_field_map(self, info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        field_map = {}
        features = info.get('features', {})
        
        for feature_name, feature_info in features.items():
            names = feature_info.get('names')
            if names:
                # Handle nested lists if present (e.g. [['j1', 'j2']])
                # The info.json shows "names": [["left_joint_1", ...]] for state/action
                # But for images it is ["height", "width", "channels"]
                
                # Flatten if it's a list of lists (common in LeRobot for state/action vectors)
                if isinstance(names, list) and len(names) > 0 and isinstance(names[0], list):
                    names = names[0] # Take the inner list

                for idx, name in enumerate(names):
                    if isinstance(name, str): # Ensure name is string
                        if name not in field_map:
                            field_map[name] = []
                        field_map[name].append((feature_name, idx))
        return field_map

    def process_parquet(self, src_path, dst_path, rule_name=None, field_map=None, mode='transform'):
        """
        Process a parquet file based on the mode and rule.
        
        Args:
            src_path (str): Input parquet path.
            dst_path (str): Output parquet path.
            rule_name (str): Name of the rule in config (e.g., 'aloha').
            field_map (dict): Field mapping from info.json.
            mode (str): 'transform', 'reverse', 'copy'.
        """
        try:
            df = pd.read_parquet(src_path)
        except Exception as e:
            print(f"Error reading {src_path}: {e}")
            return False

        if mode == 'copy':
            df.to_parquet(dst_path, engine='pyarrow')
            return True

        if mode == 'reverse':
            # Reverse the rows
            df = df.iloc[::-1].reset_index(drop=True)
            # Handle timestamp if necessary (assuming it's monotonic, we might need to fix it?
            # But usually 'timestamp' is just a column. If we reverse the video, the states 
            # are reversed in time. The timestamps should probably reflect the new time?
            # Or just be reversed values?
            # If we just reverse rows, t=0 becomes t=end.
            # Usually for training, we want t starts at 0.
            # So we might need to recalculate timestamp: t_new = t_max - t_old.
            if 'timestamp' in df.columns:
                # Assuming timestamp is scalar or 1D array
                # But df['timestamp'] might be a list of arrays? Usually it's a column.
                # Let's check if it's numeric.
                try:
                    ts = df['timestamp'].values
                    if np.issubdtype(ts.dtype, np.number):
                        max_t = ts.max()
                        df['timestamp'] = max_t - df['timestamp']
                except:
                    pass
            
            df.to_parquet(dst_path, engine='pyarrow')
            return True

        if mode == 'transform':
            if not rule_name or rule_name not in self.config.get('joint_mirror_rules', {}):
                print(f"Warning: Rule '{rule_name}' not found. Copying instead.")
                df.to_parquet(dst_path, engine='pyarrow')
                return True
            
            if not field_map:
                print("Error: Field map required for transformation.")
                return False

            replacements = self.config['joint_mirror_rules'][rule_name]
            
            # Group replacements by column
            col_replacements = {} 
            for rep in replacements:
                src_name = rep['source']
                tgt_name = rep['target']
                scale = rep.get('scale', 1.0)
                
                src_locs = field_map.get(src_name, [])
                tgt_locs = field_map.get(tgt_name, [])
                
                for s_col, s_idx in src_locs:
                    for t_col, t_idx in tgt_locs:
                        if s_col == t_col:
                            if s_col not in col_replacements:
                                col_replacements[s_col] = []
                            col_replacements[s_col].append((s_idx, t_idx, scale))

            # Apply replacements
            for col, ops in col_replacements.items():
                if col not in df.columns:
                    continue
                    
                try:
                    # df[col] is typically a Series of lists/arrays. Stack to numpy.
                    # Note: We need to handle potential variable length or other issues, 
                    # but usually LeRobot data is consistent.
                    # If it's a simple list column:
                    first_val = df[col].iloc[0]
                    if isinstance(first_val, (list, np.ndarray)):
                        arr = np.vstack(df[col].values)
                        is_sequence = True
                    else:
                        # Scalar column? Unlikely for 'joint_pos', but possible.
                        arr = df[col].values.reshape(-1, 1)
                        is_sequence = False
                except ValueError as e:
                    print(f"Skipping column {col} due to stacking error: {e}")
                    continue
                    
                new_arr = arr.copy()
                
                for src_idx, tgt_idx, scale in ops:
                    if src_idx < arr.shape[1] and tgt_idx < arr.shape[1]:
                        new_arr[:, tgt_idx] = arr[:, src_idx] * scale
                
                # Assign back
                if is_sequence:
                    df[col] = list(new_arr)
                else:
                    df[col] = new_arr.flatten()

            df.to_parquet(dst_path, engine='pyarrow')
            return True

        return False
