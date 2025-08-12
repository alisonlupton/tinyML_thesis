#!/usr/bin/env python3
"""
Dog data processing with intelligent sampling.
Separate from the CL pipeline for modularity.
"""

import pandas as pd
import numpy as np
import torch
import yaml
import os
from pathlib import Path

def load_config():
    """Load configuration from YAML file."""
    with open('dog_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['dogmove']

def identify_behavior_segments(df, min_segment_length=50):
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
    start_idx = 0
    
    # Reset index to ensure proper indexing
    df = df.reset_index(drop=True)
    
    for i in range(len(df)):
        behavior = df.iloc[i]['Selected_Behavior']
        
        if behavior != current_behavior:
            # End current segment
            if current_behavior is not None and i - start_idx >= min_segment_length:
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': i - 1,
                    'behavior': current_behavior,
                    'length': i - start_idx,
                    'start_time': df.iloc[start_idx]['t_sec'],
                    'end_time': df.iloc[i-1]['t_sec']
                })
            
            # Start new segment
            current_behavior = behavior
            start_idx = i
    
    # Handle last segment
    if current_behavior is not None and len(df) - start_idx >= min_segment_length:
        segments.append({
            'start_idx': start_idx,
            'end_idx': len(df) - 1,
            'behavior': current_behavior,
            'length': len(df) - start_idx,
            'start_time': df.iloc[start_idx]['t_sec'],
            'end_time': df.iloc[-1]['t_sec']
        })
    
    return segments

def intelligent_sample_segment(df, segment, target_samples_per_segment=100):
    """
    Intelligently sample from a behavior segment.
    
    Strategy:
    - Take samples from beginning (25%)
    - Take samples from middle (50%) 
    - Take samples from end (25%)
    """
    start_idx = segment['start_idx']
    end_idx = segment['end_idx']
    segment_length = segment['length']
    
    if segment_length <= target_samples_per_segment:
        # If segment is small, take all samples
        indices = list(range(start_idx, end_idx + 1))
    else:
        # Calculate sampling points
        n_beginning = max(1, int(target_samples_per_segment * 0.25))
        n_middle = max(1, int(target_samples_per_segment * 0.50))
        n_end = max(1, int(target_samples_per_segment * 0.25))
        
        # Beginning samples
        beginning_indices = list(range(start_idx, start_idx + n_beginning))
        
        # Middle samples (evenly spaced)
        middle_start = start_idx + (segment_length - n_middle) // 2
        middle_indices = list(range(middle_start, middle_start + n_middle))
        
        # End samples
        end_indices = list(range(end_idx - n_end + 1, end_idx + 1))
        
        indices = beginning_indices + middle_indices + end_indices
    
    return df.iloc[indices]

def intelligent_sample_dog_data(df, target_total_samples=50000, min_segment_length=50):
    """
    Intelligently sample dog data while preserving temporal structure.
    
    Args:
        df: DataFrame with dog data
        target_total_samples: Target number of samples to keep
        min_segment_length: Minimum samples to consider a behavior segment (50 samples = 1/2 second)
    
    Returns:
        Sampled DataFrame
    """
    print(f"Original data: {len(df):,} samples")
    
    # Sort by time
    df = df.sort_values(['TestNum', 't_sec'])
    
    # Identify behavior segments
    segments = identify_behavior_segments(df, min_segment_length)
    print(f"Identified {len(segments)} behavior segments")
    
    # Analyze segments
    behavior_segments = {}
    for segment in segments:
        behavior = segment['behavior']
        if behavior not in behavior_segments:
            behavior_segments[behavior] = []
        behavior_segments[behavior].append(segment)
    
    print(f"Behavior segment analysis:")
    for behavior, segs in behavior_segments.items():
        total_length = sum(seg['length'] for seg in segs)
        avg_length = total_length / len(segs)
        print(f"  {behavior}: {len(segs)} segments, {total_length:,} total samples, {avg_length:.1f} avg length")
    
    # Calculate proportional targets
    total_original_samples = sum(seg['length'] for seg in segments)
    behavior_proportions = {}
    for behavior, segs in behavior_segments.items():
        total_length = sum(seg['length'] for seg in segs)
        behavior_proportions[behavior] = total_length / total_original_samples
    
    print(f"\nBehavior proportions:")
    for behavior, prop in behavior_proportions.items():
        print(f"  {behavior}: {prop:.3f} ({prop * target_total_samples:.0f} target samples)")
    
    # Sample from each behavior proportionally
    sampled_dfs = []
    for behavior, segs in behavior_segments.items():
        target_samples = int(behavior_proportions[behavior] * target_total_samples)
        samples_per_segment = max(1, target_samples // len(segs))
        
        print(f"\nSampling {behavior}: {target_samples} total samples, ~{samples_per_segment} per segment")
        
        behavior_samples = []
        for i, segment in enumerate(segs):
            segment_df = intelligent_sample_segment(df, segment, samples_per_segment)
            behavior_samples.append(segment_df)
            
            if i < 3:  # Show first few segments
                print(f"  Segment {i+1}: {segment['length']:,} → {len(segment_df):,} samples")
        
        # Combine all segments for this behavior
        if behavior_samples:
            behavior_df = pd.concat(behavior_samples, ignore_index=True)
            sampled_dfs.append(behavior_df)
            print(f"  {behavior} total: {len(behavior_df):,} samples")
    
    # Combine all behaviors
    final_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\nFinal result: {len(final_df):,} samples")
    
    # Verify behavior distribution
    final_behavior_counts = final_df['Selected_Behavior'].value_counts()
    print(f"Final behavior distribution:")
    for behavior, count in final_behavior_counts.items():
        print(f"  {behavior}: {count:,} samples ({count/len(final_df)*100:.1f}%)")
    
    return final_df

def process_dog_data(cfg):
    """Process dog data with intelligent sampling and save results."""
    print("=== Dog Data Processing with Intelligent Sampling ===")
    
    # Load raw data
    df = pd.read_csv(cfg['cleaned_data_path'])
    print(f"Loaded {len(df):,} raw samples")
    
    # Define main behaviors (note: raw data has "Lying chest", not "Lying on chest")
    main_behaviors = [
        'Standing', 'Walking', 'Sitting', 'Lying chest', 
        'Sniffing', 'Trotting', 'Galloping'
    ]
    
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
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying chest']
    cil_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    behaviors = backbone_behaviors + cil_behaviors  # Consistent order
    behavior_to_idx = {b: i for i, b in enumerate(behaviors)}
    
    print(f"Using {len(behaviors)} behaviors: {behaviors}")
    
    # Process each dog with intelligent sampling
    processed_data = {}
    total_samples = 0
    
    for dog_id in sorted(df_filtered['DogID'].unique()):
        dog_df = df_filtered[df_filtered['DogID'] == dog_id].copy()
        
        print(f"\nProcessing Dog {dog_id}: {len(dog_df):,} raw samples")
        
        # Intelligently sample the data
        sampled_df = intelligent_sample_dog_data(dog_df, target_total_samples=50000, min_segment_length=50)
        
        if len(sampled_df) > 0:
            # Get sensor columns
            sensor_cols = []
            for col in sampled_df.columns:
                if col.startswith('ABack_') or col.startswith('ANeck_') or col.startswith('GBack_') or col.startswith('GNeck_'):
                    sensor_cols.append(col)
            
            # Extract features and labels
            X = sampled_df[sensor_cols].values.astype(np.float32)
            y = sampled_df['Selected_Behavior'].map(behavior_to_idx).values
            
            # Normalize features
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            processed_data[dog_id] = {
                'X': X,
                'y': y,
                'behaviors': sorted(sampled_df['Selected_Behavior'].unique()),
                'sensor_cols': sensor_cols,
                'sampled_df': sampled_df
            }
            
            total_samples += len(sampled_df)
            print(f"  Intelligently sampled to {len(sampled_df):,} samples")
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
        np.savez_compressed(
            dog_file,
            X=data['X'],
            y=data['y'],
            behaviors=data['behaviors'],
            sensor_cols=data['sensor_cols']
        )
        print(f"Saved Dog {dog_id} data to {dog_file}")
    
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
