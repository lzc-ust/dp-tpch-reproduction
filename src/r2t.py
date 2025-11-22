import numpy as np
import pandas as pd
import logging
from scipy.stats import laplace

logger = logging.getLogger(__name__)

class R2TMechanism:
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def exponential_mechanism(self, qualities, epsilon, T_list):
        """指数机制选择最佳T"""
        if len(qualities) == 0:
            return 0
        
        # 将质量分数转换为正值（指数机制要求）
        min_quality = min(qualities)
        adjusted_qualities = [q - min_quality + 1e-6 for q in qualities]
        
        # 计算选择概率
        probabilities = [np.exp(epsilon * q / (2 * (max(adjusted_qualities) - min(adjusted_qualities)))) 
                        for q in adjusted_qualities]
        
        # 归一化概率
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # 根据概率选择
        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        logger.info(f"指数机制选择: T={T_list[chosen_index]}, 质量分数={qualities[chosen_index]:.4f}")
        
        return chosen_index
    
    def compute_quality(self, candidate_result, ground_truth):
        """计算候选结果的质量分数（负的MSE）"""
        # 合并真实值和候选值
        merged = pd.merge(ground_truth, candidate_result, on='l_orderkey', 
                         how='inner', suffixes=('_truth', '_noisy'))
        
        if len(merged) == 0:
            return -float('inf')  # 没有共同订单，质量很差
        
        # 计算均方误差的负数（误差越小，质量越高）
        mse = ((merged['revenue'] - merged['noisy_revenue']) ** 2).mean()
        return -mse  # 返回负MSE，这样MSE越小，质量分数越高
    
    def run_mechanism(self, epsilon, T_list=None, random_state=None):
        """运行R2T机制"""
        if random_state is not None:
            np.random.seed(random_state)
        
        if T_list is None:
            # 基于数据特征设置合理的T候选值
            T_list = [1, 2, 5, 10, 20, 50]
        
        # 预算分配
        epsilon_candidate = epsilon * 0.8
        epsilon_selection = epsilon * 0.2
        
        # 获取数据
        contributions = self.data_loader.get_customer_contributions()
        ground_truth = self.data_loader.get_ground_truth()
        max_value_per_item = self.data_loader.get_max_value_per_item()
        
        logger.info(f"R2T机制开始，ε={epsilon}, 候选T={T_list}")
        
        candidate_results = []
        qualities = []
        
        for T in T_list:
            # 1. 对每个客户截断订单项
            truncated_lineitems = []
            
            for cust_key, group in contributions.groupby('c_custkey'):
                # 按贡献值降序排列，取前T个
                truncated = group.nlargest(T, 'contribution')
                truncated_lineitems.append(truncated)
            
            if not truncated_lineitems:
                continue
                
            truncated_df = pd.concat(truncated_lineitems, ignore_index=True)
            
            # 2. 在截断数据集上计算订单收入
            order_revenues = truncated_df.groupby('l_orderkey')['contribution'].sum().reset_index(name='truncated_revenue')
            
            # 3. 添加拉普拉斯噪声
            new_sensitivity = T * max_value_per_item
            scale = new_sensitivity / epsilon_candidate
            
            # 为所有可能出现在结果中的订单添加噪声
            all_orders = pd.DataFrame({'l_orderkey': ground_truth['l_orderkey']})
            order_revenues = pd.merge(all_orders, order_revenues, on='l_orderkey', how='left')
            order_revenues['truncated_revenue'] = order_revenues['truncated_revenue'].fillna(0)
            
            noise = np.random.laplace(0, scale, len(order_revenues))
            order_revenues['noisy_revenue'] = order_revenues['truncated_revenue'] + noise
            
            # 取前10个
            candidate_result = order_revenues.nlargest(10, 'noisy_revenue')
            candidate_results.append(candidate_result)
            
            # 4. 计算质量分数
            quality_score = self.compute_quality(candidate_result, ground_truth)
            qualities.append(quality_score)
            
            logger.debug(f"T={T}: 质量分数={quality_score:.4f}")
        
        if not candidate_results:
            raise ValueError("没有生成有效的候选结果")
        
        # 5. 使用指数机制选择最佳T
        best_index = self.exponential_mechanism(qualities, epsilon_selection, T_list)
        final_result = candidate_results[best_index]
        
        logger.info(f"R2T机制完成，选择T={T_list[best_index]}")
        return final_result[['l_orderkey', 'noisy_revenue']]

# 测试函数
def test_r2t():
    from data_loader import DataLoader
    from config import get_db_connection_string
    from evaluator import Evaluator
    
    loader = DataLoader(get_db_connection_string())
    mechanism = R2TMechanism(loader)
    
    # 测试运行
    result = mechanism.run_mechanism(epsilon=1.0, T_list=[1, 5, 10], random_state=42)
    print("R2T机制结果:")
    print(result)
    
    # 评估结果
    ground_truth = loader.get_ground_truth()
    evaluator = Evaluator(ground_truth)
    metrics = evaluator.evaluate_all(result, "R2T")
    print("评估结果:", metrics)

if __name__ == "__main__":
    test_r2t()