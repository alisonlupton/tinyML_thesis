#!/usr/bin/env python3
"""
Compute Continual Learning metrics from the R matrix.
Based on the mathematical definitions provided by the user.
"""

import numpy as np
import json

def calculate_cl_metrics_from_matrix(R, random_baseline=None, matrix_type="cumulative"):
    """
    Calculate CL metrics from accuracy matrix R.
    
    Args:
        R: Accuracy matrix where R[i,j] is accuracy of model trained on task i evaluated on task j
        random_baseline: Random baseline accuracies for each task (optional)
        matrix_type: "cumulative" or "new_class_only" to distinguish the type of R matrix
    
    Returns:
        Dictionary of CL metrics
    """
    K = R.shape[0]  #Number of tasks
    
    if random_baseline is None:
        #Assume random baseline is 1/num_classes for each task
        #This is a rough approximation - might want to compute actual random baselines
        random_baseline = np.full(K, 100.0 / 7)  #7 classes total
    
    metrics = {}
    
    #Average Accuracy: ACC_K = (1/K) * sum_{j=1}^K R_{K,j}
    metrics['Average_Accuracy'] = np.mean(R[-1, :])  #Last row (after training all tasks)
    
    #Average Incremental Accuracy: A_K = (2/(K(K+1))) * sum_{i>=j} R_{i,j}
    total = 0
    for i in range(K):
        for j in range(i+1):  #i >= j
            total += R[i, j]
    metrics['Average_Incremental_Accuracy'] = (2.0 / (K * (K + 1))) * total
    
    #Average Forgetting: AF_K = (1/(K-1)) * sum_{j=1}^{K-1} f^j_K
    #where f^j_K = max_{i in {1,...,K-1}} R_{i,j} - R_{K,j}
    if K > 1:
        forgetting = 0
        for j in range(K-1):  #j = 1 to K-1 (0-indexed: 0 to K-2)
            max_prev = np.max(R[:-1, j])  #max over rows 0 to K-2
            current = R[-1, j]  #current accuracy on task j
            forgetting += max_prev - current
        metrics['Average_Forgetting'] = forgetting / (K - 1)
    else:
        metrics['Average_Forgetting'] = 0.0
    
    #Intransigence: I_K = R*_K - R_{K,K}
    #where R*_K is the upper bound (perfect accuracy = 100%)
    with open("metrics_meta/oracle_baseline.json","r") as f:
        oracle = json.load(f)  #{"T1": acc1, ...}
    #assume R is KxK (no backbone row)
    K = R.shape[0]
    Rstar = np.array([oracle[f"T{i+1}"] for i in range(K)], dtype=float)
    metrics['intransigence'] = Rstar[-1] - R[-1, -1]   #papers I_K
    
    #Backward Transfer: BWT_K = (1/(K-1)) * sum_{j=1}^{K-1} (R_{K,j} - R_{j,j})
    if K > 1:
        bwt = 0
        for j in range(K-1):  #j = 1 to K-1 (0-indexed: 0 to K-2)
            bwt += R[-1, j] - R[j, j]  #final accuracy - initial accuracy on task j
        metrics['Backward_Transfer'] = bwt / (K - 1)
    else:
        metrics['Backward_Transfer'] = 0.0
    
    #Forward Transfer: FWT_K = (1/(K-1)) * sum_{j=2}^K (R_{j-1,j} - b_j)
    #where b_j is the random baseline for task j
    if K > 1:
        if matrix_type == "new_class_only":
            #For pure FWT, T1 has no previous task to transfer from, so we skip it
            #Only compute FWT for T2 and T3 (j=1,2 in 0-indexed)
            #Paper style FWT = R_{j-1,j} - b_j (using random baselines)
            fwt_values = []
            for j in range(1, K):  #j = 2 to K (0-indexed: 1 to K-1)
                #Paper style FWT = R_{j-1,j} - b_j
                fwt_values.append(R[j-1, j] - random_baseline[j])
            
            if fwt_values:
                metrics['Forward_Transfer'] = np.mean(fwt_values)
            else:
                metrics['Forward_Transfer'] = float('nan')  #No valid FWT data
        else:
            #Standard FWT calculation
            fwt = 0
            for j in range(1, K):  #j = 2 to K (0-indexed: 1 to K-1)
                fwt += R[j-1, j] - random_baseline[j]
            metrics['Forward_Transfer'] = fwt / (K - 1)
    else:
        metrics['Forward_Transfer'] = 0.0
    
    #Modified BWT (): _K = (2/(K(K-1))) * sum_{i=2}^K sum_{j=1}^{i-1} (R_{i,j} - R_{j,j})
    if K > 1:
        kappa = 0
        for i in range(1, K):  #i = 2 to K (0-indexed: 1 to K-1)
            for j in range(i):  #j = 1 to i-1 (0-indexed: 0 to i-1)
                kappa += R[i, j] - R[j, j]
        metrics['Modified_BWT_kappa'] = (2.0 / (K * (K - 1))) * kappa
    else:
        metrics['Modified_BWT_kappa'] = 0.0
    
    #Modified FWT (): _K = (2/(K(K-1))) * sum_{i<j} R_{i,j}
    if K > 1:
        zeta = 0
        for i in range(K):
            for j in range(i+1, K):  #j > i
                zeta += R[i, j]
        metrics['Modified_FWT_zeta'] = (2.0 / (K * (K - 1))) * zeta
    else:
        metrics['Modified_FWT_zeta'] = 0.0
    
    return metrics

def main():
    #Load both R matrices
    R_cumulative = np.load('R_offline.npy')
    R_new_only = np.load('R_offline_new.npy')
    
    print(f"Loaded cumulative R matrix with shape: {R_cumulative.shape}")
    print("Cumulative R matrix:")
    print(np.round(R_cumulative, 2))
    print()
    
    print(f"Loaded new-class-only R matrix with shape: {R_new_only.shape}")
    print("New-class-only R matrix:")
    print(np.round(R_new_only, 2))
    print()
    
    #Load random baseline if available
    random_baseline = None
    try:
        with open('metrics_meta/random_baseline.json', 'r') as f:
            baseline_data = json.load(f)
            #Convert task-based dict to array
            random_baseline = np.array([baseline_data[f'T{i+1}'] for i in range(R_cumulative.shape[0])])
            print(f"Loaded random baseline: {np.round(random_baseline, 2)}")
    except (FileNotFoundError, KeyError):
        print("No random baseline found, using default (100/7  14.29%)")
    
    #Compute CL metrics for cumulative matrix
    print("\n" + "="*60)
    print("CUMULATIVE R MATRIX METRICS")
    print("="*60)
    metrics_cumulative = calculate_cl_metrics_from_matrix(R_cumulative, random_baseline, "cumulative")
    
    for name, value in metrics_cumulative.items():
        print(f"{name:25}: {value:8.2f}%")
    
    #Compute CL metrics for new-class-only matrix
    print("\n" + "="*60)
    print("NEW-CLASS-ONLY R MATRIX METRICS")
    print("="*60)
    metrics_new_only = calculate_cl_metrics_from_matrix(R_new_only, random_baseline, "new_class_only")
    
    for name, value in metrics_new_only.items():
        print(f"{name:25}: {value:8.2f}%")
    
    #Save both sets of metrics
    all_metrics = {
        "cumulative": {name: round(value, 2) for name, value in metrics_cumulative.items()},
        "new_class_only": {name: round(value, 2) for name, value in metrics_new_only.items()}
    }
    
    with open('cl_metrics_from_R.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved both sets of metrics to cl_metrics_from_R.json")
    
    #Print comparison of FWT
    print("\n" + "="*60)
    print("FORWARD TRANSFER COMPARISON")
    print("="*60)
    print(f"Standard FWT (cumulative)    : {metrics_cumulative['Forward_Transfer']:8.2f}%")
    print(f"Pure FWT (new-class-only)    : {metrics_new_only['Forward_Transfer']:8.2f}%")
    print(f"Difference                   : {metrics_new_only['Forward_Transfer'] - metrics_cumulative['Forward_Transfer']:8.2f}%")

if __name__ == "__main__":
    main()
