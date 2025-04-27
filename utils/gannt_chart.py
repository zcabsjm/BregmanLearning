import matplotlib.pyplot as plt

tasks = [
    "Exploration & planning",
    "Literature review",
    "Codebase Setup",
    "LinBreg Momentum Model",
    "Initial Results Analysis",
    "Iterative Architecture Improvement (CNN)",
    "Running Experiments & Analysis",
    "Nuclear Norm Framework",
    "Running Experiments & Analysis",
    "SVD Approximation",
    "Final evaluation",
    "Dissertation Writeup",
]

# Start and duration for each task (in months, 0=October)
starts = [0, 1, 1, 1, 2, 2, 3, 3.5, 4.5, 4, 5, 4]
durations = [1, 2, 2, 1, 1, 3, 2, 2, 1, 2, 1.5, 3]

fig, ax = plt.subplots(figsize=(10, 6))

for i, (task, start, duration) in enumerate(zip(tasks, starts, durations)):
    ax.barh(i, duration, left=start, height=0.5, color="blue")

# Set y-axis
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels(tasks, fontsize=12) # Added fontsize
ax.invert_yaxis()

# Set x-axis
ax.set_xticks(range(7))
ax.set_xticklabels(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"], fontsize=12) # Added fontsize

# Grid
ax.grid(True, axis='both', linestyle=':', linewidth=0.5)

ax.set_xlabel('Month', fontsize=12) # Added fontsize
ax.set_title('Expected Timeline of the Project', fontsize=14) # Added fontsize

plt.tight_layout()
plt.show()
