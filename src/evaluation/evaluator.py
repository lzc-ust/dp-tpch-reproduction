import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import logging
import sys
import os

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
    
    def jaccard_similarity(self, noisy_result):
        """计算两个Top-10列表的Jaccard相似度"""
        truth_keys = set(self.ground_truth['l_orderkey'])
        
        # 确保noisy_result有l_orderkey列
        if 'l_orderkey' not in noisy_result.columns:
            logger.error("噪声结果中缺少l_orderkey列")
            return 0.0
            
        noisy_keys = set(noisy_result['l_orderkey'])
        
        intersection = truth_keys.intersection(noisy_keys)
        union = truth_keys.union(noisy_keys)
        
        if len(union) == 0:
            return 0.0
        
        jaccard = len(intersection) / len(union)
        return jaccard
    
    def relative_error(self, noisy_result):
        """计算总收入的相对误差"""
        truth_total = self.ground_truth['revenue'].sum()
        
        # 确保noisy_result有收入列
        if 'revenue' in noisy_result.columns:
            noisy_total = noisy_result['revenue'].sum()
        elif 'noisy_revenue' in noisy_result.columns:
            noisy_total = noisy_result['noisy_revenue'].sum()
        else:
            logger.error("噪声结果中缺少收入列")
            return float('inf')
        
        if truth_total == 0:
            return float('inf')
        
        relative_err = abs(truth_total - noisy_total) / truth_total
        return relative_err
    
    def kendall_tau(self, noisy_result):
        """计算排名的一致性（Kendall Tau）"""
        # 获取共同存在的订单
        truth_keys = set(self.ground_truth['l_orderkey'])
        
        if 'l_orderkey' not in noisy_result.columns:
            return 0.0
            
        noisy_keys = set(noisy_result['l_orderkey'])
        common_keys = truth_keys.intersection(noisy_keys)
        
        if len(common_keys) < 2:
            return 0.0
        
        # 构建共同项目的排名
        truth_rank = self.ground_truth[self.ground_truth['l_orderkey'].isin(common_keys)].copy()
        noisy_rank = noisy_result[noisy_result['l_orderkey'].isin(common_keys)].copy()
        
        # 创建排名字典
        truth_ranking = {row['l_orderkey']: i for i, (_, row) in enumerate(truth_rank.iterrows())}
        noisy_ranking = {row['l_orderkey']: i for i, (_, row) in enumerate(noisy_rank.iterrows())}
        
        # 确保顺序一致
        common_keys_list = list(common_keys)
        truth_ranks = [truth_ranking[key] for key in common_keys_list]
        noisy_ranks = [noisy_ranking[key] for key in common_keys_list]
        
        # 计算Kendall Tau
        try:
            tau, _ = kendalltau(truth_ranks, noisy_ranks)
            return tau if not np.isnan(tau) else 0.0
        except:
            return 0.0
    
    def evaluate_all(self, noisy_result, method_name=""):
        """计算所有评估指标"""
        metrics = {
            'jaccard': self.jaccard_similarity(noisy_result),
            'relative_error': self.relative_error(noisy_result),
            'kendall_tau': self.kendall_tau(noisy_result)
        }
        
        logger.debug(f"{method_name}评估结果: Jaccard={metrics['jaccard']:.3f}, "
                   f"RelativeError={metrics['relative_error']:.3f}, KendallTau={metrics['kendall_tau']:.3f}")
        
        return metrics

def run_multiple_trials(mechanism_func, epsilon, n_trials=10, method_name=""):
    """运行多次试验并计算平均指标"""
    all_metrics = []
    
    # 获取ground_truth用于评估
    # 注意：这里假设mechanism_func有一个data_loader属性
    if hasattr(mechanism_func, '__self__') and hasattr(mechanism_func.__self__, 'data_loader'):
        data_loader = mechanism_func.__self__.data_loader
        ground_truth = data_loader.get_ground_truth()
    else:
        # 如果无法获取data_loader，创建一个临时的
        from core.data_loader import DataLoader
        from utils.config import get_db_connection_string
        data_loader = DataLoader(get_db_connection_string())
        ground_truth = data_loader.get_ground_truth()
    
    evaluator = Evaluator(ground_truth)
    
    for i in range(n_trials):
        try:
            logger.info(f"运行试验 {i+1}/{n_trials}, ε={epsilon}")
            result = mechanism_func(epsilon=epsilon, random_state=i)
            
            # 评估结果
            metrics = evaluator.evaluate_all(result, f"{method_name}_trial_{i}")
            metrics['trial'] = i
            metrics['epsilon'] = epsilon
            metrics['method'] = method_name
            
            all_metrics.append(metrics)
            
        except Exception as e:
            logger.warning(f"试验 {i} 失败: {e}")
            continue
    
    if not all_metrics:
        logger.error(f"所有试验都失败了，方法: {method_name}")
        # 返回一个空的DataFrame，但包含必要的列
        return pd.DataFrame(columns=['jaccard', 'relative_error', 'kendall_tau', 'trial', 'epsilon', 'method'])
    
    return pd.DataFrame(all_metrics)

# 测试函数
def test_evaluator():
    from core.data_loader import DataLoader
    from config import get_db_connection_string
    
    # 创建测试数据
    loader = DataLoader(get_db_connection_string())
    ground_truth = loader.get_ground_truth()
    
    evaluator = Evaluator(ground_truth)
    print("黄金标准数据:")
    print(ground_truth)
    
    # 测试完美情况
    perfect_metrics = evaluator.evaluate_all(ground_truth, "Perfect")
    print("完美匹配评估:", perfect_metrics)
    
    # 测试随机数据
    random_result = ground_truth.copy()
    random_result = random_result.sample(frac=1).reset_index(drop=True)  # 随机打乱
    random_metrics = evaluator.evaluate_all(random_result, "Random")
    print("随机排序评估:", random_metrics)

if __name__ == "__main__":
    test_evaluator()