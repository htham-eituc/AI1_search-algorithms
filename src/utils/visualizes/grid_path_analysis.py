import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_path_quality(df, case_name):
    """
    Analyze path length performance.
    Separate successful vs failed algorithms.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Separate successful and failed paths
    successful = df[df['best_fitness'] != np.inf].copy()
    failed = df[df['best_fitness'] == np.inf].copy()
    
    # Plot 1: Successful algorithms only
    if len(successful) > 0:
        grouped = successful.groupby('algorithm')['best_fitness'].agg(['mean', 'std']).reset_index()
        best_path = grouped['mean'].min()
        grouped['pct_diff'] = ((grouped['mean'] - best_path) / best_path * 100).round(2)
        grouped = grouped.sort_values('pct_diff')
        
        colors = sns.color_palette("husl", len(grouped))
        bars = ax1.barh(range(len(grouped)), grouped['pct_diff'], color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (idx, row) in enumerate(grouped.iterrows()):
            label = f"{row['pct_diff']:.1f}%"
            ax1.text(row['pct_diff'] + 1, i, label, va='center', fontsize=10, fontweight='bold')
        
        ax1.set_yticks(range(len(grouped)))
        ax1.set_yticklabels(grouped['algorithm'], fontsize=10)
        ax1.set_xlabel('% Difference from Best Solution', fontsize=11, fontweight='bold')
        ax1.set_title('Successful Algorithms: Path Length Performance', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_xlim(0, grouped['pct_diff'].max() * 1.15)
    
    # Plot 2: Success rate
    algo_stats = df.groupby('algorithm').agg({
        'best_fitness': lambda x: (x != np.inf).sum()
    }).reset_index()
    algo_stats.columns = ['algorithm', 'success_count']
    algo_stats['total_runs'] = len(df) // len(df['algorithm'].unique())
    algo_stats['success_rate'] = (algo_stats['success_count'] / algo_stats['total_runs'] * 100).round(1)
    algo_stats = algo_stats.sort_values('success_rate', ascending=True)
    
    colors_success = ['#2ecc71' if x == 100 else '#e74c3c' if x == 0 else '#f39c12' 
                      for x in algo_stats['success_rate']]
    bars = ax2.barh(algo_stats['algorithm'], algo_stats['success_rate'], 
                    color=colors_success, alpha=0.8, edgecolor='black')
    
    # Add percentage labels
    for i, (idx, row) in enumerate(algo_stats.iterrows()):
        ax2.text(row['success_rate'] + 2, i, f"{row['success_rate']:.1f}%", 
                va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Algorithm Success Rate', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    fig.suptitle(f'Path Quality Analysis - {case_name}', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


# def analyze_execution_time(df, case_name):
#     """
#     Comprehensive execution time analysis with multiple views.
#     - Mean time with std error bars
#     - Time distribution comparison
#     - Successful vs failed run times
#     - Time efficiency ranking
#     """
#     fig = plt.figure(figsize=(16, 10))
#     gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
#     # Plot 1: Mean time with error bars (all runs)
#     ax1 = fig.add_subplot(gs[0, 0])
#     time_stats = df.groupby('algorithm')['execution_time_seconds'].agg(['mean', 'std']).reset_index()
#     time_stats = time_stats.sort_values('mean')
#     colors = sns.color_palette("husl", len(time_stats))
    
#     bars = ax1.barh(time_stats['algorithm'], time_stats['mean'],
#                     color=colors, alpha=0.8, edgecolor='black')
#     ax1.errorbar(time_stats['mean'], range(len(time_stats)),
#                 xerr=time_stats['std'], fmt='none',
#                 color='black', elinewidth=1.5, capsize=5)
    
#     # Add time labels
#     for i, (idx, row) in enumerate(time_stats.iterrows()):
#         ax1.text(row['mean'] * 1.1, i, f"{row['mean']:.4f}s", 
#                 va='center', fontsize=9, fontweight='bold')
    
#     ax1.set_xlabel('Execution Time (seconds)', fontsize=10, fontweight='bold')
#     ax1.set_title('Mean Execution Time (All Runs)', fontsize=11, fontweight='bold')
#     ax1.grid(axis='x', alpha=0.3, linestyle='--')
#     ax1.margins(x=0.15)
    
#     # Plot 2: Distribution box plot
#     ax2 = fig.add_subplot(gs[0, 1])
#     sns.boxplot(data=df, y='algorithm', x='execution_time_seconds',
#                ax=ax2, palette="husl", order=time_stats['algorithm'].values)
#     ax2.set_xlabel('Execution Time (seconds)', fontsize=10, fontweight='bold')
#     ax2.set_title('Time Distribution Across All Runs', fontsize=11, fontweight='bold')
#     ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
#     # Plot 3: Successful runs only
#     ax3 = fig.add_subplot(gs[1, 0])
#     successful = df[df['best_fitness'] != np.inf].copy()
#     if len(successful) > 0:
#         success_time = successful.groupby('algorithm')['execution_time_seconds'].agg(['mean', 'std']).reset_index()
#         success_time = success_time.sort_values('mean')
        
#         bars = ax3.barh(success_time['algorithm'], success_time['mean'],
#                        color=colors, alpha=0.8, edgecolor='black')
#         ax3.errorbar(success_time['mean'], range(len(success_time)),
#                     xerr=success_time['std'], fmt='none',
#                     color='black', elinewidth=1.5, capsize=5)
        
#         for i, (idx, row) in enumerate(success_time.iterrows()):
#             ax3.text(row['mean'] * 1.1, i, f"{row['mean']:.4f}s", 
#                     va='center', fontsize=9, fontweight='bold')
        
#         ax3.set_xlabel('Execution Time (seconds)', fontsize=10, fontweight='bold')
#         ax3.set_title('Mean Time - Successful Runs Only', fontsize=11, fontweight='bold')
#         ax3.grid(axis='x', alpha=0.3, linestyle='--')
#         ax3.margins(x=0.15)
    
#     # Plot 4: Violin plot for distribution detail
#     ax4 = fig.add_subplot(gs[1, 1])
#     algo_order = time_stats['algorithm'].values
#     sns.violinplot(data=df, y='algorithm', x='execution_time_seconds',
#                   ax=ax4, palette="husl", order=algo_order)
#     ax4.set_xlabel('Execution Time (seconds)', fontsize=10, fontweight='bold')
#     ax4.set_title('Time Distribution Detail (Violin Plot)', fontsize=11, fontweight='bold')
#     ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
#     fig.suptitle(f'Comprehensive Execution Time Analysis - {case_name}', 
#                 fontsize=13, fontweight='bold', y=0.995)
    
#     return fig


def analyze_optimality_gap(df, case_name):
    """
    Analyze optimality gap (difference from best solution).
    Shows both absolute and relative gaps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    successful = df[df['best_fitness'] != np.inf].copy()
    
    if len(successful) == 0:
        print(f"No successful runs for {case_name}")
        return fig
    
    # Get best known path length (assuming one of classical algorithms found it)
    best_known = successful['best_fitness'].min()
    
    # Calculate optimality gap for each algorithm
    gap_stats = successful.groupby('algorithm').agg({
        'best_fitness': ['mean', 'std', 'min']
    }).reset_index()
    gap_stats.columns = ['algorithm', 'mean_path', 'std_path', 'min_path']
    gap_stats['optimality_gap'] = gap_stats['mean_path'] - best_known
    gap_stats['gap_pct'] = (gap_stats['optimality_gap'] / best_known * 100).round(2)
    gap_stats = gap_stats.sort_values('optimality_gap')
    
    colors = sns.color_palette("husl", len(gap_stats))
    
    # Plot 1: Absolute gap
    bars1 = ax1.barh(gap_stats['algorithm'], gap_stats['optimality_gap'],
                     color=colors, alpha=0.8, edgecolor='black')
    
    for i, (idx, row) in enumerate(gap_stats.iterrows()):
        ax1.text(row['optimality_gap'] + 5, i, f"+{row['optimality_gap']:.0f}", 
                va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Optimality Gap (steps above best)', fontsize=11, fontweight='bold')
    ax1.set_title('Absolute Optimality Gap', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Plot 2: Percentage gap
    bars2 = ax2.barh(gap_stats['algorithm'], gap_stats['gap_pct'],
                     color=colors, alpha=0.8, edgecolor='black')
    
    for i, (idx, row) in enumerate(gap_stats.iterrows()):
        if row['gap_pct'] > 0:
            ax2.text(row['gap_pct'] + 1, i, f"+{row['gap_pct']:.1f}%", 
                    va='center', fontsize=10, fontweight='bold')
        else:
            ax2.text(row['gap_pct'] + 0.5, i, f"{row['gap_pct']:.1f}%", 
                    va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('% Gap from Best Solution', fontsize=11, fontweight='bold')
    ax2.set_title('Relative Optimality Gap', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    fig.suptitle(f'Optimality Gap Analysis (Best Known: {best_known:.0f} steps) - {case_name}', 
                fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig

def analyze_execution_time(df, case_name):
    """Analyze execution time across algorithms (mean only)."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_stats = df.groupby('algorithm')['execution_time_seconds'].agg(['mean', 'std']).reset_index()
    time_stats = time_stats.sort_values('mean')
    
    colors = sns.color_palette("husl", len(time_stats))
    
    ax.barh(time_stats['algorithm'], time_stats['mean'],
            color=colors, alpha=0.8, edgecolor='black')
    
    ax.errorbar(time_stats['mean'], time_stats['algorithm'],
                xerr=time_stats['std'], fmt='none',
                color='black', elinewidth=1, capsize=3)
    
    ax.set_xlabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title(f'Mean Execution Time - {case_name}', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_reliability_efficiency(df, case_name):
    """
    Scatter plot: Success Rate vs Mean Time
    Shows trade-off between reliability and speed.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    
    stats = df.groupby('algorithm').agg({
        'best_fitness': lambda x: (x != np.inf).sum() / len(x) * 100,
        'execution_time_seconds': 'mean'
    }).reset_index()
    stats.columns = ['algorithm', 'success_rate', 'mean_time']
    
    # Get colors
    colors = sns.color_palette("husl", len(stats))
    
    # Plot scatter with algorithm names as labels
    scatter = ax.scatter(stats['mean_time'], stats['success_rate'],
                        s=400, alpha=0.7, c=range(len(stats)),
                        cmap='husl', edgecolors='black', linewidth=2)
    
    # Add algorithm labels
    for idx, row in stats.iterrows():
        ax.annotate(row['algorithm'], 
                   (row['mean_time'], row['success_rate']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', alpha=0.7, facecolor='white', edgecolor='gray'))
    
    # Add quadrant lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=stats['mean_time'].median(), color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Mean Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Reliability vs Efficiency - {case_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-5, 105)
    
    # Add quadrant labels
    ax.text(0.95, 0.95, 'Fast & Reliable', transform=ax.transAxes,
           ha='right', va='top', fontsize=9, style='italic', alpha=0.6)
    ax.text(0.95, 0.05, 'Fast but Unreliable', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=9, style='italic', alpha=0.6)
    
    plt.tight_layout()
    return fig

def print_summary_stats(df, case_name):
    """
    Print comprehensive summary statistics.
    Handles both successful and failed runs.
    """
    print("\n" + "="*80)
    print(f"{case_name.upper()} - SUMMARY STATISTICS".center(80))
    print("="*80)
    
    # Overall stats
    print("\nOVERALL PERFORMANCE:")
    print("-" * 80)
    overall = df.groupby('algorithm').agg({
        'best_fitness': ['count', lambda x: (x != np.inf).sum(), 'mean', 'min'],
        'execution_time_seconds': ['mean', 'std']
    }).round(4)
    overall.columns = ['Total Runs', 'Successful', 'Mean Path', 'Best Path', 'Mean Time', 'Std Time']
    overall['Success Rate %'] = (overall['Successful'] / overall['Total Runs'] * 100).round(1)
    print(overall)
    
    # Successful runs only
    successful = df[df['best_fitness'] != np.inf]
    if len(successful) > 0:
        print("\nSUCCESSFUL RUNS ONLY:")
        print("-" * 80)
        success_stats = successful.groupby('algorithm').agg({
            'best_fitness': ['count', 'mean', 'std', 'min', 'max'],
            'execution_time_seconds': ['mean', 'std']
        }).round(4)
        success_stats.columns = ['Count', 'Mean Path', 'Std Path', 'Min Path', 'Max Path', 'Mean Time', 'Std Time']
        print(success_stats)
        
        # Optimality gap
        best_known = successful['best_fitness'].min()
        print(f"\nOPTIMALITY ANALYSIS (Best Known Path: {best_known:.0f} steps):")
        print("-" * 80)
        gap_analysis = successful.groupby('algorithm').agg({
            'best_fitness': ['min', 'mean']
        }).reset_index()
        gap_analysis.columns = ['algorithm', 'best_path', 'mean_path']
        gap_analysis['gap_from_best'] = gap_analysis['best_path'] - best_known
        gap_analysis['mean_gap'] = gap_analysis['mean_path'] - best_known
        gap_analysis['gap_pct'] = (gap_analysis['mean_gap'] / best_known * 100).round(2)
        gap_analysis = gap_analysis.sort_values('gap_pct')
        print(gap_analysis[['algorithm', 'best_path', 'mean_path', 'gap_from_best', 'gap_pct']].to_string(index=False))
    
    # Failed runs
    failed = df[df['best_fitness'] == np.inf]
    if len(failed) > 0:
        print(f"\nFAILED RUNS: {len(failed)} instances")
        print("-" * 80)
        failed_summary = failed.groupby('algorithm').agg({
            'execution_time_seconds': ['count', 'mean', 'max']
        }).round(4)
        failed_summary.columns = ['Failures', 'Mean Time', 'Max Time']
        print(failed_summary)
    
    print("\n" + "="*80 + "\n")

def analyze_nodes_expanded(df, case_name):
    """Analyze nodes expanded across algorithms with distribution.
       GA_Grid and ACO_Grid are excluded because node expansion
       is not meaningful for metaheuristics.
    """

    # Remove GA and ACO
    df = df[~df['algorithm'].isin(['GA_Grid', 'ACO_Grid'])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean nodes expanded
    stats = df.groupby('algorithm')['nodes_expanded'].agg(['mean', 'std']).reset_index()
    stats = stats.sort_values('mean', ascending=False)

    colors = sns.color_palette("husl", len(stats))

    axes[0].barh(stats['algorithm'], stats['mean'], color=colors, edgecolor='black')

    axes[0].errorbar(
        stats['mean'], stats['algorithm'],
        xerr=stats['std'],
        fmt='none',
        color='black',
        capsize=3
    )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Nodes Expanded (log scale)")
    axes[0].set_title("Mean Nodes Expanded")
    axes[0].grid(axis='x', alpha=0.3)

    # Distribution of nodes expanded
    sns.boxplot(
        data=df,
        x='algorithm',
        y='nodes_expanded',
        hue='algorithm',
        palette="husl",
        legend=False,
        ax=axes[1]
    )

    axes[1].set_yscale("log")
    axes[1].set_title("Distribution of Nodes Expanded")
    axes[1].set_xlabel("Algorithm")
    axes[1].set_ylabel("Nodes Expanded")

    fig.suptitle(
        f'Nodes Expanded Analysis - {case_name}',
        fontsize=13,
        fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

# Example usage:
"""
df = pd.read_csv('pathfinding_results.csv')
case_name = 'Grid 50x50 with obstacles'

# Generate all visualizations
fig1 = analyze_path_quality(df, case_name)
fig2 = analyze_execution_time(df, case_name)
fig3 = analyze_optimality_gap(df, case_name)
fig4 = analyze_reliability_efficiency(df, case_name)

# Print statistics
print_summary_stats(df, case_name)

# If you have detailed results with convergence curves:
# results_dict = {'GA_Grid': ga_result, 'ACO_Grid': aco_result}
# fig5 = plot_convergence_comparison(results_dict, case_name)

plt.show()
"""