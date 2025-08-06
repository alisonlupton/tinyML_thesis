# EWC + AR1* Regularization Strategies

## Overview
Added configurable EWC (Elastic Weight Consolidation) and AR1* (Adaptive Regularization) to complement latent replay for better continual learning performance.

## Configuration Options

### **EWC (Elastic Weight Consolidation)**
```yaml
EWC: True                    # Enable/disable EWC
ewc_lambda: 1000.0          # EWC regularization strength
ewc_fisher_samples: 100     # Samples for Fisher estimation
```

**How it works:**
- Computes Fisher information matrix after each task
- Prevents forgetting by penalizing large changes to important weights
- Uses diagonal approximation for efficiency

### **AR1* (Adaptive Regularization)**
```yaml
AR1: True                   # Enable/disable AR1*
ar1_lambda: 0.1             # AR1 regularization strength
ar1_alpha: 0.5              # AR1 adaptation rate
```

**How it works:**
- Adaptive L2 regularization that increases with task number
- Prevents overfitting to recent tasks
- Lightweight implementation

## Usage

### **Enable All Strategies:**
```yaml
CWR: False      # Disabled for now
EWC: True       # Enable EWC
AR1: True       # Enable AR1*
```

### **Enable Only EWC:**
```yaml
CWR: False
EWC: True
AR1: False
```

### **Enable Only AR1*:**
```yaml
CWR: False
EWC: False
AR1: True
```

### **Disable All (Baseline):**
```yaml
CWR: False
EWC: False
AR1: False
```

## Expected Benefits

### **EWC Benefits:**
- Prevents catastrophic forgetting
- Preserves important weight patterns
- Works well with replay strategies

### **AR1* Benefits:**
- Prevents overfitting to recent tasks
- Adaptive regularization strength
- Minimal computational overhead

### **Combined Benefits:**
- Better overall accuracy (closer to 0.6 target)
- More stable learning across tasks
- Reduced forgetting while maintaining learning

## Monitoring

The code will print:
- `"Using EWC"` / `"EWC turned off!"`
- `"Using AR1*"` / `"AR1* turned off!"`
- `"EWC loss: X.XXXX"` (every 50 steps)
- `"AR1* loss: X.XXXX"` (every 50 steps)
- `"Computing Fisher information for task X..."`

## Tuning Parameters

### **EWC Tuning:**
- **ewc_lambda**: Higher = stronger regularization (try 500-2000)
- **ewc_fisher_samples**: More = better Fisher estimation (try 50-200)

### **AR1* Tuning:**
- **ar1_lambda**: Higher = stronger regularization (try 0.05-0.2)
- **ar1_alpha**: Higher = faster adaptation (try 0.3-0.7)

## Next Steps

1. **Test EWC only**: Set `EWC: True, AR1: False`
2. **Test AR1* only**: Set `EWC: False, AR1: True`
3. **Test combined**: Set both to `True`
4. **Tune parameters**: Adjust lambda values based on results
5. **Compare with BNN paper**: Should achieve closer to 0.6 accuracy 