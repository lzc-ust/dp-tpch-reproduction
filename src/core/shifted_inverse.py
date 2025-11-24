# src/core/shifted_inverse.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ShiftedInverseMechanism:
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def exponential_mechanism(self, qualities, epsilon, T_list):
        """指数机制选择最佳T"""
        if len(qualities) == 0:
            return 0
        
        # 将质量分数转换为正值
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
        merged = pd.merge(ground_truth, candidate_result, on='l_orderkey', 
                         how='inner', suffixes=('_truth', '_noisy'))
        
        if len(merged) == 0:
            return -float('inf')
        
        mse = ((merged['revenue'] - merged['noisy_revenue']) ** 2).mean()
        return -mse
    
    def run_mechanism(self, epsilon, T_list=None, market_segment='BUILDING', date='1995-03-15', random_state=None):
        """运行Shifted Inverse机制"""
        if random_state is not None:
            np.random.seed(random_state)
        
        if T_list is None:
            # 基于客户贡献分布设置T候选值
            T_list = [10000, 50000, 100000, 200000, 500000]
        
        # 预算分配
        epsilon_candidate = epsilon * 0.8
        epsilon_selection = epsilon * 0.2
        
        # 获取数据
        contributions = self.data_loader.get_customer_contributions(market_segment, date)
        ground_truth = self.data_loader.get_ground_truth(market_segment, date)
        
        logger.info(f"Shifted Inverse机制开始，ε={epsilon}, 候选T={T_list}, 细分市场={market_segment}")
        
        candidate_results = []
        qualities = []
        
        for T in T_list:
            # 1. 对每个客户进行概率采样
            sampled_lineitems = []
            
            for cust_key, group in contributions.groupby('c_custkey'):
                cust_total = group['contribution'].sum()
                s = max(0, cust_total - T)
                p = 1 / (s + 1)  # 采样概率
                
                # 概率性地保留该客户的订单项
                for _, item in group.iterrows():
                    if np.random.random() < p:
                        weighted_item = item.copy()
                        weighted_item['weight'] = 1 / p
                        sampled_lineitems.append(weighted_item)
            
            if not sampled_lineitems:
                continue
                
            sampled_df = pd.DataFrame(sampled_lineitems)
            
            # 2. 在加权样本上重新聚合订单收入 - 修复groupby警告
            if not sampled_df.empty:
                # 使用agg方法避免apply的警告
                order_revenues = sampled_df.groupby('l_orderkey').agg({
                    'contribution': 'sum',
                    'weight': 'first'
                })
                order_revenues['weighted_revenue'] = order_revenues['contribution'] * order_revenues['weight']
                order_revenues = order_revenues.reset_index()[['l_orderkey', 'weighted_revenue']]
            else:
                order_revenues = pd.DataFrame(columns=['l_orderkey', 'weighted_revenue'])
            
            # 3. 添加拉普拉斯噪声
            # 敏感度是 T（经过数学推导）
            new_sensitivity = T
            scale = new_sensitivity / epsilon_candidate
            
            # 为所有可能出现在结果中的订单添加噪声
            all_orders = pd.DataFrame({'l_orderkey': ground_truth['l_orderkey']})
            if not order_revenues.empty:
                order_revenues = pd.merge(all_orders, order_revenues, on='l_orderkey', how='left')
            else:
                order_revenues = all_orders.copy()
                order_revenues['weighted_revenue'] = 0
                
            order_revenues['weighted_revenue'] = order_revenues['weighted_revenue'].fillna(0)
            
            noise = np.random.laplace(0, scale, len(order_revenues))
            order_revenues['noisy_revenue'] = order_revenues['weighted_revenue'] + noise
            
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
        
        logger.info(f"Shifted Inverse机制完成，选择T={T_list[best_index]}, 细分市场={market_segment}")
        return final_result[['l_orderkey', 'noisy_revenue']]