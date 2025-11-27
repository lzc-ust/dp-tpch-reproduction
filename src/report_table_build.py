import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Experimental results data
mechanisms = ['Naive Laplace', 'R2T', 'Shifted Inverse']
privacy_budgets = [0.1, 0.5, 1.0, 2.0]

# Revenue Error data (mechanism x privacy_budget)
revenue_errors = {
    'Naive Laplace': [0.215, 0.144, 0.080, 0.036],
    'R2T': [1.245, 0.892, 0.784, 0.635],
    'Shifted Inverse': [1.128, 0.856, 0.784, 0.702]
}

# Ranking Quality (Kendall Tau) data
ranking_quality = {
    'Naive Laplace': [0.102, 0.134, 0.156, 0.178],
    'R2T': [-0.145, -0.078, -0.022, 0.015],
    'Shifted Inverse': [0.701, 0.721, 0.733, 0.748]
}

# Market segment performance
segments = ['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE']
segment_tau = {
    'Naive Laplace': [0.142, 0.138, 0.151, 0.163, 0.145],
    'R2T': [-0.018, -0.025, -0.021, -0.023, -0.019],
    'Shifted Inverse': [0.759, 0.728, 0.741, 0.752, 0.731]
}

def create_performance_comparison_plot():
    """Create comprehensive performance comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Revenue Error plot
    x = np.arange(len(privacy_budgets))
    width = 0.25
    multiplier = 0
    
    for mechanism, errors in revenue_errors.items():
        offset = width * multiplier
        bars = ax1.bar(x + offset, errors, width, label=mechanism, alpha=0.8)
        ax1.bar_label(bars, padding=3, fmt='%.3f', fontsize=8)
        multiplier += 1
    
    ax1.set_xlabel('Privacy Budget (ε)')
    ax1.set_ylabel('Relative Error')
    ax1.set_title('Revenue Estimation Accuracy by Mechanism', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width, privacy_budgets)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Ranking Quality plot
    multiplier = 0
    for mechanism, tau_values in ranking_quality.items():
        offset = width * multiplier
        bars = ax2.bar(x + offset, tau_values, width, label=mechanism, alpha=0.8)
        ax2.bar_label(bars, padding=3, fmt='%.3f', fontsize=8)
        multiplier += 1
    
    ax2.set_xlabel('Privacy Budget (ε)')
    ax2.set_ylabel('Kendall Tau Coefficient')
    ax2.set_title('Ranking Preservation Quality by Mechanism', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width, privacy_budgets)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_market_segment_analysis():
    """Create market segment performance analysis"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(segments))
    width = 0.25
    multiplier = 0
    
    for mechanism, tau_values in segment_tau.items():
        offset = width * multiplier
        bars = ax.bar(x + offset, tau_values, width, label=mechanism, alpha=0.8)
        ax.bar_label(bars, padding=3, fmt='%.3f', fontsize=8)
        multiplier += 1
    
    ax.set_xlabel('Market Segment')
    ax.set_ylabel('Kendall Tau Coefficient')
    ax.set_title('Ranking Quality Across Market Segments (ε=1.0)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width, segments, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('market_segment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_privacy_utility_tradeoff():
    """Create privacy-utility tradeoff analysis"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mechanism in mechanisms:
        errors = revenue_errors[mechanism]
        ax.plot(privacy_budgets, errors, 'o-', linewidth=2.5, markersize=8, label=mechanism)
    
    ax.set_xlabel('Privacy Budget (ε)')
    ax.set_ylabel('Relative Error')
    ax.set_title('Privacy-Utility Tradeoff: Revenue Estimation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_parameter_tuning_impact():
    """Visualize the impact of parameter tuning on R2T performance"""
    tuning_stages = ['Before Tuning', 'After Tuning']
    error_values = [4.46, 0.635]
    tau_values = [-0.123, -0.022]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error reduction
    bars1 = ax1.bar(tuning_stages, error_values, color=['red', 'green'], alpha=0.7)
    ax1.set_ylabel('Relative Error')
    ax1.set_title('R2T Error Reduction through Parameter Tuning', fontweight='bold')
    ax1.bar_label(bars1, fmt='%.3f')
    ax1.grid(True, alpha=0.3)
    
    # Ranking improvement
    bars2 = ax2.bar(tuning_stages, tau_values, color=['red', 'green'], alpha=0.7)
    ax2.set_ylabel('Kendall Tau Coefficient')
    ax2.set_title('R2T Ranking Quality Improvement', fontweight='bold')
    ax2.bar_label(bars2, fmt='%.3f')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_tuning_impact.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_radar_chart():
    """Create radar chart for overall mechanism comparison"""
    categories = ['Revenue Accuracy', 'Ranking Quality', 'Top-k Identification', 
                 'Parameter Sensitivity', 'Consistency Across Segments']
    
    # Normalized scores (0-1, higher is better)
    naive_laplace = [0.92, 0.16, 1.00, 0.90, 0.85]  # High accuracy, low ranking
    r2t = [0.22, -0.02, 1.00, 0.15, 0.80]  # Poor overall
    shifted_inverse = [0.22, 0.73, 1.00, 0.75, 0.88]  # Excellent ranking
    
    # Adjust for radar chart (repeat first value to close the circle)
    categories = categories + [categories[0]]
    naive_laplace = naive_laplace + [naive_laplace[0]]
    r2t = r2t + [r2t[0]]
    shifted_inverse = shifted_inverse + [shifted_inverse[0]]
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True).tolist()
    
    ax.plot(angles, naive_laplace, 'o-', linewidth=2, label='Naive Laplace', markersize=6)
    ax.fill(angles, naive_laplace, alpha=0.1)
    
    ax.plot(angles, r2t, 'o-', linewidth=2, label='R2T', markersize=6)
    ax.fill(angles, r2t, alpha=0.1)
    
    ax.plot(angles, shifted_inverse, 'o-', linewidth=2, label='Shifted Inverse', markersize=6)
    ax.fill(angles, shifted_inverse, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Mechanism Performance Comparison\n(ε=1.0)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('mechanism_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

# Execute all visualizations
if __name__ == "__main__":
    print("Generating differential privacy mechanism evaluation visualizations...")
    
    create_performance_comparison_plot()
    create_market_segment_analysis()
    create_privacy_utility_tradeoff()
    create_parameter_tuning_impact()
    create_summary_radar_chart()
    
    print("All visualizations generated successfully!")
    print("Generated files:")
    print("- performance_comparison.png")
    print("- market_segment_analysis.png")
    print("- privacy_utility_tradeoff.png")
    print("- parameter_tuning_impact.png")
    print("- mechanism_radar_chart.png")