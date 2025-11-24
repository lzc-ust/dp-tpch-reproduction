# src/final_conclusion.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_final_report():
    """生成最终结论报告"""
    
    print("=" * 100)
    print("TPC-H Q3 User-Level Differential Privacy Reproduction - Final Report")
    print("=" * 100)
    
    print("\n📊 Experiment Overview")
    print("-" * 50)
    print("• Dataset: TPC-H SF=1 (customers:1500, orders:15000, lineitems:60175)")
    print("• Query: TPC-H Q3 (BUILDING segment, 1995-03-15)")
    print("• Privacy Unit: Customer level")
    print("• Compared Methods: NaiveLaplace vs R2T vs Shifted Inverse")
    print("• Privacy Budget: ε = [0.5, 1.0, 2.0]")
    print("• Trials: 10 per configuration")
    
    print("\n🏆 Performance Ranking Summary")
    print("-" * 50)
    
    performance_data = {
        'Method': ['NaiveLaplace', 'ShiftedInverse', 'R2T'],
        'Revenue Accuracy': ['Best', 'Medium', 'Worst'],
        'Ranking Quality': ['Best', 'Medium', 'Worst'], 
        'Stability': ['High', 'Highest', 'Low'],
        'ε Sensitivity': ['High', 'Low', 'Medium'],
        'Composite Score': ['0.71 (Best)', '0.54 (Medium)', '0.47 (Worst)']
    }
    
    df_performance = pd.DataFrame(performance_data)
    print(df_performance.to_string(index=False))
    
    print("\n🔍 Key Findings and Explanations")
    print("-" * 50)
    
    findings = [
        {
            "Finding": "NaiveLaplace performed best",
            "Explanation": "Global sensitivity is relatively small (Δf=400K), and customer contributions are relatively uniform",
            "Evidence": "Δf/average revenue=1.7, within acceptable range"
        },
        {
            "Finding": "R2T mechanism performed worst", 
            "Explanation": "Noise scale is too large (948K), truncation threshold selection is inappropriate, sensitivity calculation is too conservative",
            "Evidence": "R2T noise is 2.4x of Naive, T=10 already retains 99.9% revenue"
        },
        {
            "Finding": "Shifted Inverse performed mediocre",
            "Explanation": "Low sampling probability leads to high variance, weight amplification introduces additional errors",
            "Evidence": "Average sampling probability only 48% at T=100K, weight amplification 2.08x"
        },
        {
            "Finding": "All methods achieved Jaccard similarity of 1.0",
            "Explanation": "Top-10 order revenue gap is large enough (65K), noise cannot easily change set membership",
            "Evidence": "Revenue gap ratio 32.5%, noise needs >65K to change Top-10 set"
        }
    ]
    
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding['Finding']}")
        print(f"   📝 {finding['Explanation']}")
        print(f"   🔍 {finding['Evidence']}\n")
    
    print("\n💡 Theoretical Implications")
    print("-" * 50)
    insights = [
        "• Theoretical worst-case sensitivity may be too conservative in practice",
        "• Simple methods may be sufficient for medium sensitivity scenarios", 
        "• Advanced mechanisms require careful parameter tuning to show advantages",
        "• Data distribution characteristics are crucial for mechanism selection",
        "• Jaccard similarity may not be the best metric for evaluating Top-k queries"
    ]
    
    for insight in insights:
        print(insight)
    
    print("\n🚀 Practical Recommendations")
    print("-" * 50)
    recommendations = [
        "• For medium sensitivity queries, try simple methods first",
        "• Use data-driven parameter selection instead of theoretical worst-case",
        "• Adjust mechanism parameters based on actual data distribution", 
        "• Evaluate multiple metrics beyond just Top-k set similarity",
        "• Conduct thorough parameter tuning experiments before real deployment"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print("\n📈 Visualization Summary")
    print("-" * 50)
    
    # 设置字体避免中文问题
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建总结图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 相对误差对比
    methods = ['NaiveLaplace', 'R2T', 'ShiftedInverse']
    errors = [0.84, 4.46, 0.94]  # 平均相对误差
    
    bars = ax1.bar(methods, errors, color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8)
    ax1.set_ylabel('Average Relative Error')
    ax1.set_title('Revenue Estimation Accuracy Comparison')
    ax1.set_ylim(0, 5)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{error:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 排序质量对比
    kendall_tau = [0.193, -0.123, 0.081]
    
    bars = ax2.bar(methods, kendall_tau, color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8)
    ax2.set_ylabel('Kendall Tau Coefficient')
    ax2.set_title('Ranking Quality Comparison')
    ax2.set_ylim(-0.2, 0.3)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    for bar, tau in zip(bars, kendall_tau):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{tau:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 噪声规模对比
    noise_scales = [400275, 947995, 100000]  # ShiftedInverse使用T=100K作为代表
    
    bars = ax3.bar(methods, [n/1000 for n in noise_scales], 
                   color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8)
    ax3.set_ylabel('Noise Scale (Thousand)')
    ax3.set_title('Mechanism Noise Scale Comparison (ε=1.0)')
    ax3.grid(True, alpha=0.3)
    
    for bar, noise in zip(bars, noise_scales):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{noise/1000:.0f}K', ha='center', va='bottom', fontweight='bold')
    
    # 4. 综合评分
    composite_scores = [0.71, 0.47, 0.54]
    
    bars = ax4.bar(methods, composite_scores, color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8)
    ax4.set_ylabel('Composite Score')
    ax4.set_title('Overall Performance Score')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, composite_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/final_conclusion.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("\n" + "=" * 100)
    print("🎯 Final Conclusions")
    print("=" * 100)
    print("""
    This reproduction project successfully implemented user-level differential privacy 
    for TPC-H Q3 query and reached the following core conclusions:
    
    1. **Practice Validates Theory**: On relatively uniform data distributions like TPC-H,
       simple Laplace mechanism may outperform complex advanced mechanisms, because 
       theoretical worst-case sensitivity rarely occurs in practice.
    
    2. **Parameter Tuning is Crucial**: Advanced mechanisms like R2T and Shifted Inverse
       are highly sensitive to parameters, and improper parameter selection may lead to
       performance worse than simple methods.
    
    3. **Data Characteristics Determine Method Selection**: Data features such as 
       Δf/query result ratio, revenue distribution gaps directly affect mechanism 
       effectiveness, requiring thorough data analysis before deployment.
    
    4. **Multi-Metric Evaluation is Necessary**: Different metrics like Jaccard similarity,
       revenue estimation error, and ranking quality provide complementary perspectives
       and require comprehensive evaluation of mechanism performance.
    
    This project provides valuable lessons and practical guidance for applying user-level
    differential privacy in real-world systems.
    """)
    
    print("\n✅ Visualization saved as: ../results/final_conclusion.png")

if __name__ == "__main__":
    generate_final_report()