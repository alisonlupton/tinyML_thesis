"""
Example script showing how to integrate task accuracy tracking into your training loop.
This demonstrates how to track accuracy on all tasks as new tasks are learned.
"""

import torch
import torch.nn as nn
import numpy as np
from model_learning_debugger import ModelLearningDebugger

def example_task_accuracy_tracking():
    """Example of how to track task accuracies during continual learning"""
    
    # Initialize the debugger
    debugger = ModelLearningDebugger()
    
    # Simulate training across multiple tasks
    num_tasks = 5
    epochs_per_task = 2
    steps_per_epoch = 100
    
    print("🚀 Starting Task Accuracy Tracking Example")
    
    for task_id in range(num_tasks):
        print(f"\n=== Training Task {task_id} ===")
        
        for epoch in range(epochs_per_task):
            print(f"  Epoch {epoch}")
            
            for step in range(steps_per_epoch):
                # Simulate training step (replace with your actual training)
                # This is where you'd normally do forward/backward pass
                
                # Simulate accuracy for current task
                current_task_accuracy = 0.5 + 0.4 * (step / steps_per_epoch) + 0.1 * np.random.random()
                
                # Track accuracy for current task
                debugger.track_task_accuracy(
                    task_id=task_id,
                    epoch=epoch,
                    step=step,
                    accuracy=current_task_accuracy,
                    task_name=f"Task_{task_id}"
                )
                
                # Simulate evaluating all previous tasks (continual learning scenario)
                for prev_task in range(task_id + 1):
                    # Simulate forgetting: accuracy decreases as new tasks are learned
                    forgetting_factor = max(0.1, 1.0 - 0.1 * (task_id - prev_task))
                    prev_task_accuracy = (0.8 + 0.1 * np.random.random()) * forgetting_factor
                    
                    debugger.track_task_accuracy(
                        task_id=prev_task,
                        epoch=epoch,
                        step=step,
                        accuracy=prev_task_accuracy,
                        task_name=f"Task_{prev_task}"
                    )
            
            # Generate task accuracy report after each epoch
            if epoch == epochs_per_task - 1:  # Last epoch of the task
                print(f"  📊 Generating task accuracy report for Task {task_id}")
                debugger.generate_task_accuracy_report(current_task_id=task_id)
    
    print(f"\n✅ Task accuracy tracking completed!")
    print(f"📁 Check the plots in: {debugger.save_dir}/plots/")

def integration_with_main_training():
    """
    Example of how to integrate this into your main training loop.
    Add these lines to your main.py where you evaluate tasks.
    """
    
    # Initialize debugger (do this once at the start)
    debugger = ModelLearningDebugger()
    
    # In your training loop, after evaluating each task:
    """
    # After training on task t, evaluate all tasks 0 to t
    for eval_task in range(t + 1):
        # Your evaluation code here
        eval_accuracy = evaluate_task(model, eval_task, test_loader)
        
        # Track the accuracy
        debugger.track_task_accuracy(
            task_id=eval_task,
            epoch=current_epoch,
            step=current_step,
            accuracy=eval_accuracy,
            task_name=f"Task_{eval_task}"
        )
    
    # Generate report after task t is complete
    if t == num_tasks - 1:  # After last task
        debugger.generate_task_accuracy_report(current_task_id=t)
    """
    
    print("📝 Integration example shown in comments above")

if __name__ == "__main__":
    print("🎯 Task Accuracy Tracking Example")
    print("=" * 50)
    
    # Run the example
    example_task_accuracy_tracking()
    
    print("\n" + "=" * 50)
    print("📝 Integration Instructions:")
    integration_with_main_training()
