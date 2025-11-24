# src/adaptive_mechanisms.py
import pandas as pd
import numpy as np
import logging
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.data_loader import DataLoader
from config import get_db_connection_string

logger = logging.getLogger(__name__)

class AdaptiveParameterSelector:
    """自适应参数选择器"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def analyze_data_distribution(self, contributions):
        """分析数据分布特征"""
        stats = {}
        
        # 客户级统计
        customer_stats = contributions.groupby('c_custkey').agg({
            'contribution': ['sum', 'count'],
            'l_extendedprice': 'mean'
        })
        
        stats['customer_count'] = len(customer_stats)
        stats['max_customer_contribution'] = customer_stats[('contribution', 'sum')].max()
        stats['avg_customer_contribution'] = customer_stats[('contribution', 'sum')].mean()
        stats['median_customer_contribution'] = customer_stats[('contribution', 'sum')].median()
        
        # 订单项数量分布
        item_counts = customer_stats[('contribution', 'count')]
        stats['max_items_per_customer'] = item_counts.max()
        stats['avg_items_per_customer'] = item_counts.mean()
        stats['item_count_quantiles'] = item_counts.quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
        
        # 贡献值分布
        contribution_sums = customer_stats[('contribution', 'sum')]
        stats['contribution_quantiles'] = contribution_sums.quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
        
        return stats
    
    def suggest_r2t_parameters(self, stats):
        """为R2T机制推荐参数"""
        item_quantiles = stats['item_count_quantiles']
        
        # 基于订单项数量的分位数选择T值
        T_candidates = [
            int(item_quantiles[0.25]),  # 25%分位数
            int(item_quantiles[0.5]),   # 中位数
            int(item_quantiles[0.75]),  # 75%分位数
            int(item_quantiles[0.9])    # 90%分位数
        ]
        
        # 去除重复和过小的值
        T_candidates = sorted(list(set([max(1, t) for t in T_candidates])))
        
        logger.info(f"R2T参数建议: T_candidates = {T_candidates}")
        return T_candidates
    
    def suggest_shifted_inverse_parameters(self, stats):
        """为Shifted Inverse机制推荐参数"""
        contribution_quantiles = stats['contribution_quantiles']
        
        # 基于客户贡献的分位数选择T值
        T_candidates = [
            int(contribution_quantiles[0.25]),  # 25%分位数
            int(contribution_quantiles[0.5]),   # 中位数  
            int(contribution_quantiles[0.75]),  # 75%分位数
            int(contribution_quantiles[0.9])    # 90%分位数
        ]
        
        # 确保T值合理
        T_candidates = sorted(list(set([max(1000, t) for t in T_candidates])))
        
        logger.info(f"Shifted Inverse参数建议: T_candidates = {T_candidates}")
        return T_candidates

class AdaptiveDPMechanism:
    """自适应差分隐私机制"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.parameter_selector = AdaptiveParameterSelector(data_loader)
    
    def run_adaptive_r2t(self, epsilon, random_state=None):
        """运行自适应R2T机制"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # 获取当前查询条件下的数据
        contributions = self.data_loader.get_customer_contributions()
        
        # 分析数据分布
        stats = self.parameter_selector.analyze_data_distribution(contributions)
        
        # 获取推荐的参数
        T_candidates = self.parameter_selector.suggest_r2t_parameters(stats)
        
        # 运行R2T机制
        from r2t import R2TMechanism
        mechanism = R2TMechanism(self.data_loader)
        result = mechanism.run_mechanism(epsilon=epsilon, T_list=T_candidates, random_state=random_state)
        
        return result, stats, T_candidates
    
    def run_adaptive_shifted_inverse(self, epsilon, random_state=None):
        """运行自适应Shifted Inverse机制"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # 获取当前查询条件下的数据
        contributions = self.data_loader.get_customer_contributions()
        
        # 分析数据分布
        stats = self.parameter_selector.analyze_data_distribution(contributions)
        
        # 获取推荐的参数
        T_candidates = self.parameter_selector.suggest_shifted_inverse_parameters(stats)
        
        # 运行Shifted Inverse机制
        from shifted_inverse import ShiftedInverseMechanism
        mechanism = ShiftedInverseMechanism(self.data_loader)
        result = mechanism.run_mechanism(epsilon=epsilon, T_list=T_candidates, random_state=random_state)
        
        return result, stats, T_candidates

def test_adaptive_mechanisms():
    """测试自适应机制"""
    loader = DataLoader(get_db_connection_string())
    adaptive_mech = AdaptiveDPMechanism(loader)
    
    print("测试自适应机制...")
    
    # 测试自适应R2T
    result, stats, T_candidates = adaptive_mech.run_adaptive_r2t(epsilon=1.0, random_state=42)
    print(f"自适应R2T结果: {len(result)}条记录")
    print(f"数据统计: {stats['customer_count']}个客户")
    print(f"推荐的T值: {T_candidates}")
    
    # 评估结果
    from evaluator import Evaluator
    ground_truth = loader.get_ground_truth()
    evaluator = Evaluator(ground_truth)
    metrics = evaluator.evaluate_all(result, "Adaptive_R2T")
    print(f"评估指标: {metrics}")

if __name__ == "__main__":
    test_adaptive_mechanisms()