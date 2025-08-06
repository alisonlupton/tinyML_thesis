# Critical Bug Fix: New Classes Not Learning

## The Problem
The model was learning Task 0 well (70% accuracy) but completely failing to learn Task 1 (only 0.6% accuracy). The model was predicting mostly class 1 (from Task 0) even for Task 1 data.

## Root Cause
The issue was in the CWR* implementation. After training each task, the code was overwriting ALL task class weights with the CWR bank values. However, for new classes, the CWR bank contained only small random initialization values, effectively resetting the newly learned weights.

### The Bug:
```python
# This was overwriting newly learned weights for new classes!
for c in task_classes:
    frozen_model.classifier.weight[c].copy_(frozen_model.classifier.cwr_bank[c])
```

## The Fix
Only overwrite weights with CWR bank for classes that have been properly trained before:

```python
for c in task_classes:
    # Only overwrite if this class has been properly trained before
    if frozen_model.classifier.cwr_bank[c].norm() > 1e-3:  # Threshold to check if properly trained
        print(f"Restoring class {c} from CWR bank")
        frozen_model.classifier.weight[c].copy_(frozen_model.classifier.cwr_bank[c])
    else:
        print(f"Keeping trained weights for new class {c}")
```

## Additional Improvements

1. **Reduced replay in first epoch**: Less replay buffer usage in the first epoch to focus on learning new classes
2. **Increased learning rate**: From 0.01 to 0.02 for better learning of new classes
3. **Added weight decay**: 1e-4 for regularization
4. **Added debugging**: Track logits and predictions for current task classes

## Expected Results
With this fix, the model should now:
- Learn new classes properly in each task
- Maintain knowledge of previous tasks (CWR* working correctly)
- Show balanced predictions across all classes
- Achieve much higher accuracy on new tasks

## Testing
Run the training again and you should see:
- Task 1 accuracy > 50% (instead of 0.6%)
- Balanced predictions across classes 0,1,2,3
- Better overall continual learning performance 