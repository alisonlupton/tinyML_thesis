#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

#Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, load_processed_dog_data, set_seed, make_global_to_local_map, ClassRegistry
from models.cnn import SimplifiedTDMModelCNN
from cnn_CIL_training_utils import load_CIL_data, check_CIL_validity
from cnn_backbone_training_utils import load_data_cnn_backbone, train_cnn_backbone
import random
import torch.nn as nn

def extract_class_features(model, data_loader, class_names, device):
    """Extract backbone features for each class and compute mean features."""
    model.eval()
    features_by_class = {name: [] for name in class_names}
    
    print("Extracting features for each class...")
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(data_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            #Get backbone features (before the head)
            features = model.backbone(xb)  #Shape: [batch_size, channels, time_steps]
            #Flatten features for similarity computation
            features = features.view(features.size(0), -1)  #Shape: [batch_size, channels*time_steps]
            
            #Group features by class
            for i, class_name in enumerate(class_names):
                mask = (yb == i)
                if mask.any():
                    class_features = features[mask]
                    features_by_class[class_name].append(class_features.cpu())
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx}")
    
    #Compute mean features for each class
    mean_features = {}
    for class_name, feature_list in features_by_class.items():
        if feature_list:
            all_features = torch.cat(feature_list, dim=0)
            mean_features[class_name] = all_features.mean(dim=0)
            print(f"  {class_name}: {len(all_features)} samples, mean feature shape: {mean_features[class_name].shape}")
        else:
            print(f"  WARNING: No samples found for {class_name}")
            mean_features[class_name] = None
    
    return mean_features

def compute_class_similarities(mean_features):
    """Compute cosine similarity between all class pairs."""
    similarities = {}
    class_names = list(mean_features.keys())
    
    print("\nComputing cosine similarities between classes...")
    for i, class1 in enumerate(class_names):
        for j, class2 in enumerate(class_names):
            if i < j:  #Avoid duplicates and self-similarity
                if mean_features[class1] is not None and mean_features[class2] is not None:
                    sim = F.cosine_similarity(
                        mean_features[class1].unsqueeze(0), 
                        mean_features[class2].unsqueeze(0)
                    ).item()
                    similarities[f"{class1}-{class2}"] = sim
                    print(f"  {class1} - {class2}: {sim:.4f}")
    
    return similarities

def create_similarity_matrix(mean_features, class_names):
    """Create a similarity matrix for visualization."""
    n_classes = len(class_names)
    similarity_matrix = np.zeros((n_classes, n_classes))
    
    for i, class1 in enumerate(class_names):
        for j, class2 in enumerate(class_names):
            if mean_features[class1] is not None and mean_features[class2] is not None:
                sim = F.cosine_similarity(
                    mean_features[class1].unsqueeze(0), 
                    mean_features[class2].unsqueeze(0)
                ).item()
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  #Symmetric
            else:
                similarity_matrix[i, j] = 0.0
                similarity_matrix[j, i] = 0.0
    
    return similarity_matrix

def plot_similarity_heatmap(similarity_matrix, class_names, save_path=None):
    """Create a heatmap of class similarities."""
    plt.figure(figsize=(10, 8))
    
    #Create heatmap
    sns.heatmap(similarity_matrix, 
                xticklabels=class_names, 
                yticklabels=class_names,
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                cbar_kws={'label': 'Cosine Similarity'})
    
    plt.title('Feature Similarity Between Classes', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Classes', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Similarity heatmap saved to {save_path}")
    
    plt.show()

def analyze_sniffing_similarities(similarities, class_names):
    """Analyze similarities specifically involving Sniffing."""
    print("\n" + "="*50)
    print("ANALYSIS: Sniffing Similarities")
    print("="*50)
    
    sniffing_similarities = {}
    
    #Find all similarities involving Sniffing
    for pair, sim in similarities.items():
        if "Sniffing" in pair:
            other_class = pair.replace("Sniffing-", "").replace("-Sniffing", "")
            sniffing_similarities[other_class] = sim
    
    #Sort by similarity
    sorted_similarities = sorted(sniffing_similarities.items(), key=lambda x: x[1], reverse=True)
    
    print("Sniffing's similarity to other classes (highest to lowest):")
    for class_name, sim in sorted_similarities:
        print(f"  {class_name}: {sim:.4f}")
    
    #Compare with other new classes
    print("\nComparing new classes (Sniffing, Trotting, Galloping):")
    new_classes = ["Sniffing", "Trotting", "Galloping"]
    backbone_classes = ["Standing", "Walking", "Sitting", "Lying chest"]
    
    for new_class in new_classes:
        print(f"\n{new_class} similarities to backbone classes:")
        new_class_similarities = []
        for backbone_class in backbone_classes:
            #Try both orderings of the pair
            pair_key1 = f"{new_class}-{backbone_class}"
            pair_key2 = f"{backbone_class}-{new_class}"
            
            sim = None
            if pair_key1 in similarities:
                sim = similarities[pair_key1]
            elif pair_key2 in similarities:
                sim = similarities[pair_key2]
            
            if sim is not None:
                new_class_similarities.append(sim)
                print(f"  {backbone_class}: {sim:.4f}")
        
        if new_class_similarities:
            avg_sim = np.mean(new_class_similarities)
            print(f"  Average similarity to backbone: {avg_sim:.4f}")

def main():
    print("Feature Similarity Analysis for Dog Behaviors")
    print("=" * 50)
    
    #Load config and set device
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #Set seed for reproducibility (exactly like training script)
    seed = cfg['random_seed']
    set_seed(seed)
    gen_seed = torch.Generator()
    gen_seed.manual_seed(seed)
    
    #Load data (exactly like training script)
    print("\nLoading data...")
    dog_data, behavior_to_idx, behaviors = load_processed_dog_data()
    
    if len(dog_data) == 0:
        print("No dog data loaded!")
        return
    
    #Get dimensions (exactly like training script)
    all_behaviors = cfg['main_behaviors']
    backbone_behaviors = cfg['backbone_behaviors']
    backbone_num_classes = len(backbone_behaviors)
    C = len(dog_data[list(dog_data.keys())[0]]['sensor_cols'])
    L = int(dog_data[list(dog_data.keys())[0]]['window_len'])
    
    #Choose target dog (exactly like training script)
    all_possible_dogs = sorted(dog_data.keys())
    valid_CIL_dogs_for_this_seed = [
        dog_id for dog_id in all_possible_dogs 
        if check_CIL_validity(dog_data, dog_id, all_behaviors, behavior_to_idx, seed)
    ]
    
    rng = random.Random(seed)  
    target_dog = rng.choice(valid_CIL_dogs_for_this_seed)
    print(f"Target Dog Chosen: {target_dog}")
    
    #Choose validation dog and backbone dogs (exactly like training script)
    remaining = sorted([d for d in all_possible_dogs if d != target_dog])
    validation_dog = rng.choice(remaining)
    print(f"Validation Dog Chosen: {validation_dog}")
    
    remaining = sorted([dog for dog in all_possible_dogs if dog not in [target_dog, validation_dog]])
    backbone_dogs = rng.sample(remaining, cfg['num_pretrain_dogs'])
    print(f"Backbone Dogs Chosen: {backbone_dogs}")
    
    #Load backbone data (exactly like training script)
    backbone_data = load_data_cnn_backbone(backbone_dogs, dog_data, behavior_to_idx, backbone_behaviors, validation_dog)
    
    #Load CIL data (exactly like training script)
    X_train, y_train, X_test, y_test = load_CIL_data(dog_data, target_dog, all_behaviors, behavior_to_idx, backbone_data.train_mean, backbone_data.train_std, seed, verbose=True)
    
    #Create class registry and remap labels (exactly like training script)
    backbone_registry = ClassRegistry(all_behaviors)
    m_test, gids_all = make_global_to_local_map(all_behaviors, behavior_to_idx, device=y_test.device)
    y_test_remapped = m_test[y_test]
    
    #Create data loader (exactly like training script)
    test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
    
    #Create backbone model (exactly like training script)
    print("\nCreating backbone model...")
    backbone_model = SimplifiedTDMModelCNN(C, backbone_num_classes, device, sparsity_ratio=cfg['backbone_sparsity_ratio'])
    backbone_model.to(device)
    
    #Create data loaders for backbone training
    backbone_train_loader = DataLoader(TensorDataset(backbone_data.X_train, backbone_data.y_train), 
                                     batch_size=cfg['backbone_batch_size_train'], shuffle=True, generator=gen_seed)
    backbone_val_loader = DataLoader(TensorDataset(backbone_data.X_val, backbone_data.y_val), 
                                   batch_size=cfg['backbone_batch_size_val'], shuffle=False)
    
    #Create optimizer and criterion
    backbone_optimizer = torch.optim.Adam(backbone_model.parameters(), lr=cfg['backbone_learning_rate'], weight_decay=cfg['backbone_weight_decay'])
    backbone_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    #Train backbone model
    print("Training backbone model for feature analysis...")
    backbone_model, _ = train_cnn_backbone(backbone_model, backbone_behaviors, backbone_optimizer, backbone_criterion, backbone_train_loader, backbone_val_loader, device, cfg)
    
    backbone_model.eval()
    
    #Extract features for each class
    mean_features = extract_class_features(backbone_model, test_loader, all_behaviors, device)
    
    #Compute similarities
    similarities = compute_class_similarities(mean_features)
    
    #Create similarity matrix
    similarity_matrix = create_similarity_matrix(mean_features, all_behaviors)
    
    #Plot heatmap
    plot_similarity_heatmap(similarity_matrix, all_behaviors, 
                           save_path=os.path.join(cfg['plot_dir'], 'feature_similarity_heatmap.pdf'))
    
    #Analyze Sniffing specifically
    analyze_sniffing_similarities(similarities, all_behaviors)
    
    #Save results to CSV
    results_df = pd.DataFrame(list(similarities.items()), columns=['Class_Pair', 'Cosine_Similarity'])
    results_df = results_df.sort_values('Cosine_Similarity', ascending=False)
    
    csv_path = os.path.join(cfg['plot_dir'], 'feature_similarities.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSimilarity results saved to {csv_path}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
