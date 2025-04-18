import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
import datetime as dt

def create_gantt_chart(tasks, fig_size=(12, 8), colors=None, title="Project Timeline", max_weeks=28):
    """
    Create a Gantt chart for project visualization with weeks on x-axis.
    
    Parameters:
    -----------
    tasks : list of dict
        List of task dictionaries with keys:
        - 'name': Task name
        - 'start': Start day (days from project start)
        - 'duration': Task duration in days
        - 'status': Task status for color coding ('scheduled-unused', 'scheduled-used', 'late', 'extension')
        - 'completion': Completion percentage (optional, 0-100)
        - 'deliverable': Optional deliverable text (no longer displayed)
    fig_size : tuple
        Figure size (width, height)
    colors : dict
        Dict mapping status to colors
    title : str
        Chart title
    max_weeks : int
        Maximum number of weeks to display on x-axis
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Set font to Times New Roman for all text elements
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Default colors for the requested status categories
    if colors is None:
        colors = {
            'scheduled-unused': '#add8e6',  # Light blue
            'scheduled-used': '#1f77b4',    # Dark blue
            'late': '#d62728',              # Red
            'extension': '#2ca02c',         # Green
            'default': '#7f7f7f'            # Gray for unspecified status
        }
    
    # Sort tasks by start date
    tasks = sorted(tasks, key=lambda x: x['start'])
    
    # Create figure and axis with squared background
    fig, ax = plt.subplots(figsize=fig_size, facecolor='#f0f0f0')
    ax.set_facecolor('#f0f0f0')
    
    # Track the y positions and labels
    labels = []
    y_positions = []
    
    # Convert days to weeks for display (7 days per week)
    days_to_weeks = lambda days: days / 7
    
    # Plot each task as a horizontal bar
    for i, task in enumerate(tasks):
        labels.append(task['name'])
        y_positions.append(i)
        
        # Convert days to weeks for plotting
        start_week = days_to_weeks(task['start'])
        duration_weeks = days_to_weeks(task['duration'])
        
        status = task.get('status', 'default')
        color = colors.get(status, colors['default'])
        
        # Add the main task bar
        ax.barh(i, duration_weeks, left=start_week, color=color, 
                height=0.5, alpha=0.8, edgecolor='black')
        
        # Add completion indicator if available
        if 'completion' in task and task['completion'] > 0:
            completion_width = duration_weeks * (task['completion'] / 100)
            ax.barh(i, completion_width, left=start_week, color=color, 
                    height=0.5, alpha=1.0)
    
    # Set up the axes
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(labels)
    
    # Set up week-based x-axis
    ax.set_xlabel('Weeks from Project Start')
    ax.set_xlim(0, max_weeks)
    ax.set_xticks(range(0, max_weeks + 1, 2))  # Every 2 weeks
    
    # Set title in bold
    ax.set_title(title, fontweight='bold', fontsize=14)
    
    # Add grid lines for squared background look
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create legend for status categories
    legend_elements = [
        Patch(facecolor=colors['scheduled-unused'], label='Scheduled (unused)'),
        Patch(facecolor=colors['scheduled-used'], label='Scheduled (used)'),
        Patch(facecolor=colors['late'], label='Late'),
        Patch(facecolor=colors['extension'], label='Extension')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add border around the plot area
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # Add vertical "Deliverables" title on the left side
    fig.text(0.02, 0.5, 'Deliverables', rotation=90, fontsize=14, 
             fontweight='bold', va='center', ha='center')
    
    # Tight layout with padding to accommodate the vertical title
    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Left padding for vertical title
    
    return fig, ax


def demo_gantt_chart():
    """
    Create a sample Gantt chart with tasks and the new color coding.
    """
    # Sample tasks for a project
    tasks = [
        {
            'name': 'Background Readings', 
            'start': 0, 
            'duration': 21, 
            'status': 'scheduled-used', 
            'completion': 100,
            'deliverable': 'Literature Summary'
        },
        {
            'name': 'Problem Definition', 
            'start': 14, 
            'duration': 10, 
            'status': 'scheduled-used', 
            'completion': 100
        },
        {
            'name': 'Data Collection', 
            'start': 21, 
            'duration': 14, 
            'status': 'scheduled-used', 
            'completion': 90,
            'deliverable': 'Dataset'
        },
        {
            'name': 'Algorithm Design', 
            'start': 28, 
            'duration': 21, 
            'status': 'scheduled-used', 
            'completion': 75,
            'deliverable': 'Design Document'
        },
        {
            'name': 'Implementation', 
            'start': 42, 
            'duration': 28, 
            'status': 'scheduled-used', 
            'completion': 60
        },
        {
            'name': 'Unit Testing', 
            'start': 56, 
            'duration': 14, 
            'status': 'late', 
            'completion': 40,
            'deliverable': 'Test Report'
        },
        {
            'name': 'Integration Testing', 
            'start': 63, 
            'duration': 14, 
            'status': 'late', 
            'completion': 20
        },
        {
            'name': 'Documentation', 
            'start': 70, 
            'duration': 21, 
            'status': 'scheduled-unused', 
            'completion': 10,
            'deliverable': 'User Manual'
        },
        {
            'name': 'Progress Meetings', 
            'start': 0, 
            'duration': 84, 
            'status': 'scheduled-used', 
            'completion': 60
        },
        {
            'name': 'Final Report', 
            'start': 77, 
            'duration': 28, 
            'status': 'extension', 
            'completion': 0,
            'deliverable': 'Final Report'
        }
    ]
    
    fig, ax = create_gantt_chart(tasks, title="Project Progress", max_weeks=28)
    plt.savefig('project_gantt_chart.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demo_gantt_chart()