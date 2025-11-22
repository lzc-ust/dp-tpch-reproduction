import numpy as np
import pandas as pd
import logging
from scipy.stats import laplace

logger = logging.getLogger(__name__)

class NaiveLaplaceMechanism:
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def calculate_global_sensitivity(self):
        """计算全局敏感度Δf - 一个客户能产生的最大收入影响"""
        contributions = self.data_loader.get_customer_contributions()
        
        # 计算每个客户的总贡献
        customer_totals = contributions.groupby('c_custkey')['contribution'].sum()
        delta_f = customer_totals.max()
        
        logger.info(f"计算得到全局敏感度 Δf = {delta_f:.2f}")
        return delta_f
    
    def run_mechanism(self, epsilon, random_state=None):
        """运行朴素拉普拉斯机制"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # 获取黄金标准数据
        ground_truth = self.data_loader.get_ground_truth()
        
        # 计算全局敏感度
        delta_f = self.calculate_global_sensitivity()
        
        # 对每个订单的收入添加拉普拉斯噪声
        noisy_result = ground_truth.copy()
        scale = delta_f / epsilon
        
        # 添加噪声
        noise = np.random.laplace(0, scale, len(noisy_result))
        noisy_result['noisy_revenue'] = noisy_result['revenue'] + noise
        
        # 根据噪声收入重新排序并取前10
        final_result = noisy_result.nlargest(10, 'noisy_revenue')
        
        logger.info(f"朴素拉普拉斯机制完成，ε={epsilon}")
        return final_result[['l_orderkey', 'noisy_revenue', 'o_orderdate', 'o_shippriority']]

# 测试函数
def test_naive_laplace():
    from data_loader import DataLoader
    
    # 使用你的数据库连接信息
    loader = DataLoader("mysql+pymysql://root:123456@localhost:3306/tpc_h")
    mechanism = NaiveLaplaceMechanism(loader)
    
    # 测试运行
    result = mechanism.run_mechanism(epsilon=1.0, random_state=42)
    print("朴素拉普拉斯机制结果:")
    print(result)
    
    # 测试敏感度计算
    delta_f = mechanism.calculate_global_sensitivity()
    print(f"全局敏感度: {delta_f}")

if __name__ == "__main__":
    test_naive_laplace()