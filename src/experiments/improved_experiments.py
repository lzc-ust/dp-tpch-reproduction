import pandas as pd
import numpy as np
import logging
import sys
import os

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.data_loader import DataLoader
from core.r2t import R2TMechanism
from core.shifted_inverse import ShiftedInverseMechanism
from evaluation.evaluator import run_multiple_trials
from utils.config import get_db_connection_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedExperimentRunner:
    def __init__(self):
        self.loader = DataLoader(get_db_connection_string())
    
    def run_improved_r2t(self, epsilon=1.0, n_trials=10):
        """使用改进参数的R2T实验"""
        print("运行改进的R2T实验...")
        
        # 基于数据特征选择更好的T值
        contributions = self.loader.get_customer_contributions()
        customer_counts = contributions.groupby('c_custkey').size()
        
        # 选择基于分位数的T值
        T_candidates = [
            int(customer_counts.quantile(0.5)),  # 中位数
            int(customer_counts.quantile(0.75)), # 75%分位数
            int(customer_counts.quantile(0.9)),  # 90%分位数
        ]
        
        print(f"改进的T候选值: {T_candidates}")
        
        mechanism = R2TMechanism(self.loader)
        results = run_multiple_trials(
            lambda **kwargs: mechanism.run_mechanism(T_list=T_candidates, **kwargs),
            epsilon, n_trials, "Improved_R2T"
        )
        
        return results
    
    def run_improved_shifted_inverse(self, epsilon=1.0, n_trials=10):
        """使用改进参数的Shifted Inverse实验"""
        print("运行改进的Shifted Inverse实验...")
        
        contributions = self.loader.get_customer_contributions()
        customer_totals = contributions.groupby('c_custkey')['contribution'].sum()
        
        # 基于客户贡献分布选择T值
        T_candidates = [
            int(customer_totals.quantile(0.5)),   # 中位数
            int(customer_totals.quantile(0.75)),  # 75%分位数
            int(customer_totals.quantile(0.9)),   # 90%分位数
        ]
        
        print(f"改进的T候选值: {T_candidates}")
        
        mechanism = ShiftedInverseMechanism(self.loader)
        results = run_multiple_trials(
            lambda **kwargs: mechanism.run_mechanism(T_list=T_candidates, **kwargs),
            epsilon, n_trials, "Improved_SI"
        )
        
        return results

def main():
    runner = ImprovedExperimentRunner()
    
    # 运行改进实验
    improved_r2t = runner.run_improved_r2t(epsilon=1.0, n_trials=5)
    improved_si = runner.run_improved_shifted_inverse(epsilon=1.0, n_trials=5)
    
    # 对比结果
    if not improved_r2t.empty:
        print(f"改进R2T平均误差: {improved_r2t['relative_error'].mean():.3f}")
    if not improved_si.empty:
        print(f"改进SI平均误差: {improved_si['relative_error'].mean():.3f}")

if __name__ == "__main__":
    main()