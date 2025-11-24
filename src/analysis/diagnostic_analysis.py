import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.data_loader import DataLoader
from utils.config import get_db_connection_string
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosticAnalyzer:
    def __init__(self):
        self.loader = DataLoader(get_db_connection_string())
    
    def analyze_data_characteristics(self):
        """分析数据特征以解释实验结果"""
        print("=" * 80)
        print("数据特征诊断分析")
        print("=" * 80)
        
        # 获取基础数据
        ground_truth = self.loader.get_ground_truth()
        contributions = self.loader.get_customer_contributions()
        
        print("\n1. 黄金标准数据分析:")
        print("-" * 40)
        print(f"Top-10订单总收入: {ground_truth['revenue'].sum():.2f}")
        print(f"单个订单平均收入: {ground_truth['revenue'].mean():.2f}")
        print(f"收入范围: {ground_truth['revenue'].min():.2f} - {ground_truth['revenue'].max():.2f}")
        
        # 分析收入分布
        revenue_gap = ground_truth['revenue'].iloc[0] - ground_truth['revenue'].iloc[9]
        print(f"第1名与第10名收入差距: {revenue_gap:.2f}")
        print(f"收入差距比例: {revenue_gap / ground_truth['revenue'].iloc[9] * 100:.1f}%")
        
        print("\n2. 客户贡献分析:")
        print("-" * 40)
        customer_totals = contributions.groupby('c_custkey')['contribution'].sum()
        print(f"总客户数: {len(customer_totals)}")
        print(f"最大客户贡献: {customer_totals.max():.2f}")
        print(f"平均客户贡献: {customer_totals.mean():.2f}")
        print(f"客户贡献中位数: {customer_totals.median():.2f}")
        
        # 分析客户贡献分布
        top_customers = customer_totals.nlargest(5)
        print(f"前5大客户贡献: {list(top_customers.round(2))}")
        
        print("\n3. 全局敏感度分析:")
        print("-" * 40)
        delta_f = customer_totals.max()
        max_item_value = self.loader.get_max_value_per_item()
        print(f"全局敏感度 Δf: {delta_f:.2f}")
        print(f"单个订单项最大价值: {max_item_value:.2f}")
        print(f"Δf / 平均订单收入: {delta_f / ground_truth['revenue'].mean():.2f}")
        
        # 分析噪声规模
        epsilon = 1.0
        scale_naive = delta_f / epsilon
        scale_r2t = 10 * max_item_value / epsilon  # 假设T=10
        print(f"\nε=1.0时的噪声规模:")
        print(f"NaiveLaplace噪声规模: {scale_naive:.2f}")
        print(f"R2T噪声规模 (T=10): {scale_r2t:.2f}")
        print(f"噪声/平均收入比例 - Naive: {scale_naive / ground_truth['revenue'].mean():.2f}")
        print(f"噪声/平均收入比例 - R2T: {scale_r2t / ground_truth['revenue'].mean():.2f}")
        
        print("\n4. 关键洞察:")
        print("-" * 40)
        if revenue_gap > scale_naive:
            print("✓ 收入差距 > 噪声规模 → 排序容易保持正确")
        else:
            print("✗ 收入差距 < 噪声规模 → 排序容易被打乱")
            
        if delta_f / ground_truth['revenue'].mean() < 10:
            print("✓ 敏感度相对较小 → NaiveLaplace可行")
        else:
            print("✗ 敏感度很大 → NaiveLaplace噪声过大")
    
    def analyze_mechanism_behavior(self):
        """分析各机制的具体行为"""
        print("\n" + "=" * 80)
        print("机制行为分析")
        print("=" + "-" * 79)
        
        contributions = self.loader.get_customer_contributions()
        customer_totals = contributions.groupby('c_custkey')['contribution'].sum()
        
        print("\nR2T机制问题诊断:")
        print("-" * 40)
        
        # 分析R2T的截断效应
        T_values = [1, 5, 10, 20]
        for T in T_values:
            truncated_revenue = 0
            for cust_key, group in contributions.groupby('c_custkey'):
                truncated = group.nlargest(T, 'contribution')
                truncated_revenue += truncated['contribution'].sum()
            
            original_revenue = contributions['contribution'].sum()
            loss_pct = (1 - truncated_revenue / original_revenue) * 100
            print(f"T={T}: 保留 {truncated_revenue/original_revenue*100:.1f}% 收入, 损失 {loss_pct:.1f}%")
        
        print("\nShifted Inverse机制问题诊断:")
        print("-" * 40)
        
        # 分析Shifted Inverse的采样效应
        T_values = [10000, 50000, 100000]
        for T in T_values:
            total_prob = 0
            count = 0
            for cust_key, group in contributions.groupby('c_custkey'):
                cust_total = group['contribution'].sum()
                s = max(0, cust_total - T)
                p = 1 / (s + 1)
                total_prob += p
                count += 1
            
            avg_prob = total_prob / count if count > 0 else 0
            print(f"T={T}: 平均采样概率 {avg_prob:.4f}")

def main():
    analyzer = DiagnosticAnalyzer()
    analyzer.analyze_data_characteristics()
    analyzer.analyze_mechanism_behavior()

if __name__ == "__main__":
    main()