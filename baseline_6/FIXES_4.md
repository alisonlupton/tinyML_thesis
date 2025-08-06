# Conservative Approach for Stability

## Problems Identified:
1. **Complete forgetting**: Task 0 → 0% after Task 1
2. **Poor new class learning**: Low accuracy on new tasks
3. **Replay overwhelming learning**: Replay loss (2.5+) >> Live loss (1.3)
4. **Unstable predictions**: Model predicts mostly one class

## Conservative Fixes Applied:

### 1. **Minimal Replay Strategy**
- **Epoch 0**: 0% replay (NO replay - focus entirely on new classes)
- **Epoch 1**: 10% replay (minimal retention)
- **Epoch 2**: 20% replay (small retention)

### 2. **Conservative Loss Weighting**
- **Live data**: Full weight (1.0)
- **Replay data**: 10% weight (0.1) - heavily reduced from 0.5

### 3. **Reduced Replay Batch Size**
- **From**: 50 samples
- **To**: 20 samples

### 4. **Lower Learning Rate**
- **From**: 0.02
- **To**: 0.01

### 5. **Learning Rate Scheduler**
- **Step size**: 1 epoch
- **Gamma**: 0.8 (reduce LR by 20% each epoch)

### 6. **Disabled CWR***
- **CWR**: False (temporarily disabled to isolate replay issues)

## Expected Results:

### **Task 0 (Classes 0,1):**
- Should achieve ~70%+ accuracy initially
- Should maintain >30% accuracy after Task 1 (some retention)

### **Task 1 (Classes 2,3):**
- Should achieve >60% accuracy (like without replay)
- Should show balanced predictions

### **Overall Stability:**
- Lower replay loss values
- More balanced predictions
- Gradual learning progression

## Key Principle:
**"Learn new classes first, then worry about retention"**

This approach prioritizes learning new classes in the first epoch, then gradually introduces minimal replay to prevent complete forgetting.

## Monitoring:
- **Replay loss**: Should be much lower than before
- **Live loss**: Should dominate the total loss
- **Task accuracy**: Should be more stable across tasks
- **Predictions**: Should be more balanced 