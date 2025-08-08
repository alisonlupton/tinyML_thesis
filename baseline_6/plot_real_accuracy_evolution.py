import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Extract accuracy data from the terminal output
# Format: [Task X] acc: Y.YYY
accuracy_data = {
    'Task 0': [0.837, 0.232, 0.377, 0.404],  # After tasks 0, 1, 2, 3
    'Task 1': [None, 0.461, 0.204, 0.205],   # After tasks 1, 2, 3 (None for before task 1)
    'Task 2': [None, None, 0.358, 0.212],    # After tasks 2, 3 (None for before task 2)
    'Task 3': [None, None, None, 0.164]      # After task 3 (None for before task 3)
}

# Create the plot
plt.figure(figsize=(12, 8))

# Color palette for tasks
colors = ['blue', 'green', 'orange', 'red']
markers = ['o', 's', '^', 'D']

# Plot each task's accuracy evolution
for i, (task_name, accuracies) in enumerate(accuracy_data.items()):
    # Filter out None values and create x-axis positions
    valid_accuracies = [(j, acc) for j, acc in enumerate(accuracies) if acc is not None]
    
    if valid_accuracies:
        x_positions, y_accuracies = zip(*valid_accuracies)
        
        plt.plot(x_positions, y_accuracies, 
                color=colors[i], 
                marker=markers[i], 
                linewidth=2, 
                markersize=8,
                label=task_name,
                alpha=0.8)

# Customize the plot
plt.title('Task Accuracy Evolution ', 
          fontsize=16, fontweight='bold')
plt.xlabel('Tasks Completed', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)

# Add task labels on x-axis
plt.xticks([0, 1, 2, 3], ['After Task 0', 'After Task 1', 'After Task 2', 'After Task 3'])

# Add annotations for key observations
# plt.annotate('Task 0 peaks at 83.7%', xy=(0, 0.837), xytext=(0.5, 0.9),
#             arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
#             fontsize=10, color='blue')

# plt.annotate('Catastrophic forgetting\nvisible after Task 1', xy=(1, 0.232), xytext=(1.5, 0.3),
#             arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
#             fontsize=10, color='red')

plt.tight_layout()

# Save the plot
plt.savefig('logs/real_task_accuracy_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Real task accuracy evolution plot saved: logs/real_task_accuracy_evolution.png")

# Print summary statistics
print("\n📊 REAL TASK ACCURACY SUMMARY:")
for task_name, accuracies in accuracy_data.items():
    valid_accuracies = [acc for acc in accuracies if acc is not None]
    if valid_accuracies:
        final_acc = valid_accuracies[-1]
        max_acc = max(valid_accuracies)
        min_acc = min(valid_accuracies)
        print(f"   {task_name}: Final={final_acc:.3f}, Max={max_acc:.3f}, Min={min_acc:.3f}")

# Create a summary table
print("\n📋 ACCURACY MATRIX:")
print("Task\tAfter T0\tAfter T1\tAfter T2\tAfter T3")
print("-" * 50)
for i, (task_name, accuracies) in enumerate(accuracy_data.items()):
    row = f"T{i}\t"
    for acc in accuracies:
        if acc is not None:
            row += f"{acc:.3f}\t\t"
        else:
            row += "N/A\t\t"
    print(row)
