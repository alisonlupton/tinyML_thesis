# process_dog_data.py

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from utils import load_config

def identify_behavior_segments(df, min_segment_length):
    """
    Identify continuous segments of the same behavior.
    
    Args:
        df: DataFrame with 'Selected_Behavior' and 't_sec' columns
        min_segment_length: Minimum samples to consider a segment
    
    Returns:
        List of segment dictionaries with start_idx, end_idx, behavior, length
    """
    segments = []

    current_behavior = None
    current_testnum = None
    start_idx = 0
    

    for i in range(len(df)):
        behavior = df.iloc[i]['Selected_Behavior']
        test_num = df.iloc[i]['TestNum']
        
        if behavior != current_behavior or test_num != current_testnum:
            # End current segment
            if current_behavior is not None and i - start_idx >= min_segment_length:
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': i - 1,
                    'behavior': current_behavior,
                    'test_num': current_testnum,
                    'length': i - start_idx,
                    'start_time': df.iloc[start_idx]['t_sec'],
                    'end_time': df.iloc[i-1]['t_sec']
                })
            
            # Start new segment
            current_behavior = behavior
            current_testnum = test_num
            start_idx = i
    
    # Handle last segment
    if current_behavior is not None and len(df) - start_idx >= min_segment_length:
        segments.append({
            'start_idx': start_idx,
            'end_idx': len(df) - 1,
            'behavior': current_behavior,
            'test_num': current_testnum,
            'length': len(df) - start_idx,
            'start_time': df.iloc[start_idx]['t_sec'],
            'end_time': df.iloc[-1]['t_sec']
        })
    
    return segments


def compute_window_params(cfg):
    """Derive window size (L) and stride (S) in samples from config."""
    fs = cfg['sample_rate_hz']
    win_sec = cfg['window_seconds']
    stride_sec = cfg['stride_seconds']
    L = int(round(fs * win_sec))
    S = int(round(fs * stride_sec))
    if L <= 0 or S <= 0:
        raise ValueError(f"Invalid window/stride: L={L}, S={S} (fs={fs}, win={win_sec}, stride={stride_sec})")
    return L, S, fs


def windowize_segment(df, dog_id, seg, sensor_cols, L, S, behavior_to_idx):
    """
    df: full (unthinned) per-dog dataframe, sorted by ['TestNum','t_sec']
    seg: dict with start_idx/end_idx etc. produced by identify_behavior_segments, get one dict with one call of this func 
    Returns: list of (Xw, yw, seg_id, (DogID, TestNum))
    """
    start = seg['start_idx']
    end   = seg['end_idx']
    length = end - start + 1
    if length < L:
        return []

    # CONTIGUOUS slice of the original df for this whole segment (one segment!)
    seg_df = df.iloc[start:end+1]

    # compute window starts in absolute row space (no gaps)
    starts = list(range(start, end - L + 2, S))
    # tail coverage
    last_start = end - L + 1
    if starts[-1] != last_start:
        starts.append(last_start)

    # pull contiguous array once
    arr = seg_df[sensor_cols].to_numpy(dtype=np.float32)  # (length, C)
    y_seg = behavior_to_idx[seg['behavior']]
    testn = int(seg_df['TestNum'].iloc[0])

    out = []
    for ws in starts:
        s = ws - start
        e = s + L
        Xw = arr[s:e]  # (L, C)
        out.append((Xw, y_seg, None, (dog_id, testn)))  # seg_id filled later
        
    # list of windows for this single behaviour segment for a single dog
    # Xw is list of features (sensor readings) for a single window
    return out

def make_windows_for_dog_FULL(df, dog_id, segments, behavior_to_idx, sensor_cols, cfg):
    L, S, fs = compute_window_params(cfg)

    # ensure sorted + contiguous indexing is done in Main

    all_X, all_y, all_seg, all_sess = [], [], [], []
    next_seg_id = 0
    windows_per_segment = []

    for seg in segments:
        wlist = windowize_segment(df, dog_id, seg, sensor_cols, L, S, behavior_to_idx)
        if not wlist:
            continue
        seg_id = next_seg_id
        next_seg_id += 1

        # attach seg_id to each window
        for (Xw, y, _, sess) in wlist:
            all_X.append(Xw)
            all_y.append(y)
            all_seg.append(seg_id)
            all_sess.append([sess[0], sess[1]])
        windows_per_segment.append((seg_id, seg['behavior'], len(wlist)))

    # stack features per window

    X_w = np.stack(all_X, axis=0)  # (N, L, C)
    y_w = np.asarray(all_y, dtype=np.int64)
    seg_id_w = np.asarray(all_seg, dtype=np.int32)
    session_id_w = np.asarray(all_sess, dtype=np.int32)

    # flatten for MLP, keep shape for CNN 
    if cfg['flatten_windows']:
        N, L_, C = X_w.shape
        X_w = X_w.reshape(N, L_ * C)

    return X_w, y_w, seg_id_w, session_id_w, L, S


##---------- MAIN FUNCTION 
def process_dog_data(cfg):
    """Process dog data with intelligent sampling and save results."""
    print("=== Dog Data Processing with Intelligent Sampling ===")
    
    # Load raw data
    df = pd.read_csv(cfg['cleaned_data_path'])
    print(f"Loaded {len(df):,} raw samples")
    
    # Define main behaviors 
    main_behaviors = cfg['main_behaviors']
    
    # Select behavior function
    def select_behavior(row):
        behaviors = [row['Behavior_1'], row['Behavior_2'], row['Behavior_3']]
        for behavior in behaviors:
            if behavior != '<undefined>' and behavior in main_behaviors:
                return behavior
        return None
    
    df['Selected_Behavior'] = df.apply(select_behavior, axis=1)
    df_filtered = df[df['Selected_Behavior'].notna()].copy()
    df_filtered = df_filtered.drop(["Behavior_1", "Behavior_2", "Behavior_3"], axis=1)
    
    print(f"Filtered to {len(df_filtered):,} samples with valid behaviors")
    
    # Create behavior mapping - use consistent order for CIL
    backbone_behaviors = cfg['backbone_behaviors']
    cil_behaviors = cfg['cil_behaviors']
    behaviors = backbone_behaviors + cil_behaviors  # Consistent order
    behavior_to_idx = {b: i for i, b in enumerate(behaviors)}
    
    print(f"Using {len(behaviors)} behaviors: {behaviors}")
    
    # Process each dog with intelligent sampling
    processed_data = {}
    total_samples = 0
    
    for dog_id in sorted(df_filtered['DogID'].unique()):
        dog_df = df_filtered[df_filtered['DogID'] == dog_id].copy()
        dog_df = dog_df.sort_values(['TestNum','t_sec']).reset_index(drop=True)
        
        print(f"\nProcessing Dog {dog_id}: {len(dog_df):,} raw samples")
        
        # Intelligently sample the data
        segments = identify_behavior_segments(dog_df, cfg['min_segment_length'])
        
        if len(segments) > 0:
            
            # Get sensor columns
            sensor_cols = [c for c in dog_df.columns if c.startswith(('ABack_','ANeck_','GBack_','GNeck_'))]
            sensor_cols = sorted(sensor_cols)  # ensures consistent ordering across dogs
            
            # Extract features and labels (RAW DATA - no normalization)
            # build fixed windows that inherit segment_id
            X_w, y_w, seg_id_w, session_id_w, L, S = make_windows_for_dog_FULL(
                dog_df, dog_id, segments, behavior_to_idx, sensor_cols, cfg
            )

            processed_data[dog_id] = {
                'X': X_w,                          # (N, L*C) if flattened, else (N, L, C)
                'y': y_w,
                'segment_ids': seg_id_w,
                'session_ids': session_id_w,       # (DogID, TestNum)
                'behaviors': sorted(dog_df['Selected_Behavior'].unique()),
                'sensor_cols': sensor_cols,
                'window_len': int(L),
                'stride': int(S),
                'sample_rate': cfg['sample_rate_hz']
            }

            
            total_samples += X_w.shape[0]
            print(f" Windows for dog {dog_id}: {X_w.shape[0]:,}")
        else:
            print(f"  No valid samples created")
    
    print(f"\nTotal samples processed: {total_samples:,}")
    
    # Analyze sample distribution
    all_behaviors = []
    for dog_id, data in processed_data.items():
        all_behaviors.extend([behaviors[y] for y in data['y']])
    
    behavior_counts = pd.Series(all_behaviors).value_counts()
    print(f"\nOverall behavior distribution:")
    for behavior, count in behavior_counts.items():
        print(f"  {behavior}: {count:,} samples")
    
    # Save processed data
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save each dog's data separately
    for dog_id, data in processed_data.items():
        dog_file = output_dir / f"dog_{dog_id}_intelligent.npz"
        # save *.npz — replace old keys
        np.savez_compressed(
            output_dir / f"dog_{dog_id}_intelligent.npz",
            X=processed_data[dog_id]['X'],
            y=processed_data[dog_id]['y'],
            segment_ids=processed_data[dog_id]['segment_ids'],
            session_ids=processed_data[dog_id]['session_ids'],
            behaviors=processed_data[dog_id]['behaviors'],
            sensor_cols=processed_data[dog_id]['sensor_cols'],
            window_len=processed_data[dog_id]['window_len'],
            stride=processed_data[dog_id]['stride'],
            sample_rate=processed_data[dog_id]['sample_rate']
        )
        print(f"Saved Dog {dog_id} data to {dog_file}")
        
        
        
        counts_records = []
        for dog_id, data in processed_data.items():
            y_vals = data['y']
            unique, counts = np.unique(y_vals, return_counts=True)
            for cls, cnt in zip(unique, counts):
                counts_records.append({
                    'dog_id': dog_id,
                    'behavior': behaviors[cls],
                    'behavior_idx': int(cls),
                    'count': int(cnt)
                })

        counts_df = pd.DataFrame(counts_records)
        counts_csv_path = output_dir / "dog_sample_counts.csv"
        counts_df.to_csv(counts_csv_path, index=False)
        print(f"Saved per-dog sample counts to {counts_csv_path}")

    
    # Save metadata
    metadata = {
        'behavior_to_idx': behavior_to_idx,
        'behaviors': behaviors,
        'total_dogs': len(processed_data),
        'total_samples': total_samples,
        'behavior_counts': behavior_counts.to_dict()
    }
    
    metadata_file = output_dir / "intelligent_sampling_metadata.yaml"
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"\nSaved metadata to {metadata_file}")
    
    # Create summary file
    summary_file = output_dir / "intelligent_sampling_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=== Intelligent Sampling Data Processing Summary ===\n\n")
        f.write(f"Total dogs processed: {len(processed_data)}\n")
        f.write(f"Total samples: {total_samples:,}\n")
        f.write(f"Behaviors: {behaviors}\n\n")
        
        f.write("Per-dog sample counts:\n")
        for dog_id, data in processed_data.items():
            f.write(f"  Dog {dog_id}: {len(data['X']):,} samples\n")
        
        f.write(f"\nBehavior distribution:\n")
        for behavior, count in behavior_counts.items():
            f.write(f"  {behavior}: {count:,} samples ({count/total_samples*100:.1f}%)\n")
    
    print(f"Saved summary to {summary_file}")
    
    return processed_data, behavior_to_idx, behaviors

def main():
    """Main processing function."""
    print("Starting dog data processing with intelligent sampling...")
    
    # Load configuration
    cfg = load_config()
    
    # Process data
    processed_data, behavior_to_idx, behaviors = process_dog_data(cfg)
    
    print(f"\n=== Processing Complete ===")
    print(f"Processed {len(processed_data)} dogs")
    print(f"Total samples: {sum(len(data['X']) for data in processed_data.values()):,}")
    print(f"Data saved to processed_data/ directory")
    print(f"Ready for CL pipeline to load!")

if __name__ == "__main__":
    main()
