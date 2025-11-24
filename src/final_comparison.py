# src/final_comparison.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.data_loader import DataLoader
from utils.config import get_db_connection_string

def create_final_comparison():
    """创建原始方法与改进方法的最终对比"""
    
    print("=" * 80)
    print("原始方法与改进方法最终对比分析")
    print("=" * 80)
    
    # 原始实验结果
    original_results = {
        'Method': ['NaiveLaplace', 'R2T', 'ShiftedInverse'],
        'Original_Error': [0.84, 4.46, 0.94],
        'Improved_Error': [0.84, 0.635, 0.937],  # NaiveLaplace保持不变
        'Improvement_Pct': [0, (4.46-0.635)/4.46*100, (0.94-0.937)/0.94*100]
    }
    
    df = pd.DataFrame(original_results)
    df['Improvement_Pct'] = df['Improvement_Pct'].round(1)
    
    print("\n📊 误差对比表:")
    print("-" * 50)
    print(df.to_string(index=False))
    
    print("\n🔍 关键发现:")
    print("-" * 50)
    print("1. R2T机制通过参数调优实现了巨大改进:")
    print(f"   • 误差从 {4.46:.3f} 降低到 {0.635:.3f}")
    print(f"   • 改进幅度: {((4.46-0.635)/4.46*100):.1f}%")
    print(f"   • 现在成为最佳方法!")
    
    print("\n2. Shifted Inverse机制改进有限:")
    print(f"   • 误差从 {0.94:.3f} 略微降低到 {0.937:.3f}")
    print(f"   • 改进幅度: {((0.94-0.937)/0.94*100):.1f}%")
    
    print("\n3. 新的性能排名:")
    print("   • 第1名: R2T (改进后) - 误差: 0.635")
    print("   • 第2名: NaiveLaplace - 误差: 0.840") 
    print("   • 第3名: ShiftedInverse - 误差: 0.937")
    
    # 创建对比图表
    plt.figure(figsize=(12, 8))
    
    # 设置位置
    x = np.arange(len(df['Method']))
    width = 0.35
    
    # 创建柱状图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. 误差对比
    bars1 = ax1.bar(x - width/2, df['Original_Error'], width, 
                   label='Original', color='red', alpha=0.7)
    bars2 = ax1.bar(x + width/2, df['Improved_Error'], width, 
                   label='Improved', color='green', alpha=0.7)
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Average Relative Error')
    ax1.set_title('Original vs Improved Methods: Revenue Estimation Error')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Method'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 改进百分比
    bars3 = ax2.bar(x, df['Improvement_Pct'], 
                   color=['gray', 'blue', 'gray'], alpha=0.7)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Improvement Percentage (%)')
    ax2.set_title('Improvement Percentage by Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Method'])
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, improvement in zip(bars3, df['Improvement_Pct']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{improvement:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 添加零线
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/final_improvement_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 最终结论:")
    print("-" * 50)
    print("""
    通过参数调优实验，我们得出了更完整的结论：
    
    1. **参数调优的重要性得到验证**：R2T机制通过数据驱动的参数选择
       从最差方法变成了最佳方法，误差降低了85.7%。
    
    2. **不同机制对参数调优的敏感性不同**：
       • R2T: 高度敏感，参数选择至关重要
       • Shifted Inverse: 相对不敏感，改进有限  
       • NaiveLaplace: 无参数需要调优
    
    3. **实际部署建议**：
       • 对于R2T类机制，必须进行充分的参数调优
       • 使用数据分布特征（分位数）来指导参数选择
       • 在简单方法和复杂方法之间权衡调优成本
    
    这个发现强调了在实际应用中，机制选择和参数调优应该基于具体数据特征，
    而不是单纯依赖理论分析。
    """)
    
    print(f"\n✅ 对比图表已保存: ../results/final_improvement_comparison.png")

if __name__ == "__main__":
    create_final_comparison()