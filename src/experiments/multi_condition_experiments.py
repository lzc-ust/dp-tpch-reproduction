# src/multi_condition_experiments.py (最终修复版本)
import pandas as pd
import numpy as np
import logging
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.data_loader import DataLoader
from utils.config import get_db_connection_string
from evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiConditionExperiment:
    """多查询条件实验"""
    
    def __init__(self):
        self.loader = DataLoader(get_db_connection_string())
    
    def run_experiment_for_condition(self, condition_name, custom_loader=None):
        """为特定查询条件运行实验"""
        if custom_loader is None:
            custom_loader = self.loader
        
        print(f"\n{'='*60}")
        print(f"运行实验: {condition_name}")
        print(f"{'='*60}")
        
        # 获取当前条件的黄金标准
        ground_truth = custom_loader.get_ground_truth()
        contributions = custom_loader.get_customer_contributions()
        
        print(f"数据特征:")
        print(f"  • 客户数量: {contributions['c_custkey'].nunique()}")
        print(f"  • 订单项数量: {len(contributions)}")
        print(f"  • Top-10订单总收入: {ground_truth['revenue'].sum():.2f}")
        
        # 分析敏感度
        customer_totals = contributions.groupby('c_custkey')['contribution'].sum()
        delta_f = customer_totals.max()
        print(f"  • 全局敏感度 Δf: {delta_f:.2f}")
        print(f"  • Δf/平均收入: {delta_f/ground_truth['revenue'].mean():.2f}")
        
        results = []
        
        # 测试各种机制
        mechanisms = [
            ('NaiveLaplace', self._run_naive),
            ('Adaptive_R2T', self._run_adaptive_r2t),
            ('Adaptive_SI', self._run_adaptive_si)
        ]
        
        for method_name, mechanism_func in mechanisms:
            try:
                # 直接调用函数
                result = mechanism_func(custom_loader, epsilon=1.0, random_state=42)
                evaluator = Evaluator(ground_truth)
                metrics = evaluator.evaluate_all(result, method_name)
                
                # 确保包含所有必要字段
                metrics['method'] = method_name  # 添加method列
                metrics['condition'] = condition_name
                metrics['delta_f'] = delta_f
                results.append(metrics)
                
                print(f"  {method_name}: 误差={metrics['relative_error']:.3f}, Tau={metrics['kendall_tau']:.3f}")
                
            except Exception as e:
                print(f"  {method_name} 失败: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def _run_naive(self, loader, epsilon, random_state):
        """运行朴素拉普拉斯机制"""
        from core.naive_laplace import NaiveLaplaceMechanism
        mechanism = NaiveLaplaceMechanism(loader)
        return mechanism.run_mechanism(epsilon=epsilon, random_state=random_state)
    
    def _run_adaptive_r2t(self, loader, epsilon, random_state):
        """运行自适应R2T"""
        from core.r2t import R2TMechanism
        
        # 使用自适应参数选择
        contributions = loader.get_customer_contributions()
        
        # 简单基于数据特征推荐参数
        customer_stats = contributions.groupby('c_custkey').agg({
            'contribution': ['sum', 'count']
        })
        item_counts = customer_stats[('contribution', 'count')]
        
        # 基于分位数选择T值
        quantiles = item_counts.quantile([0.25, 0.5, 0.75])
        T_candidates = [max(1, int(q)) for q in quantiles]
        
        mechanism = R2TMechanism(loader)
        return mechanism.run_mechanism(epsilon=epsilon, T_list=T_candidates, random_state=random_state)
    
    def _run_adaptive_si(self, loader, epsilon, random_state):
        """运行自适应Shifted Inverse"""
        from core.shifted_inverse import ShiftedInverseMechanism
        
        # 使用自适应参数选择
        contributions = loader.get_customer_contributions()
        
        # 简单基于数据特征推荐参数
        customer_stats = contributions.groupby('c_custkey').agg({
            'contribution': ['sum', 'count']
        })
        customer_totals = customer_stats[('contribution', 'sum')]
        
        # 基于分位数选择T值
        quantiles = customer_totals.quantile([0.25, 0.5, 0.75])
        T_candidates = [max(1000, int(q)) for q in quantiles]
        
        mechanism = ShiftedInverseMechanism(loader)
        return mechanism.run_mechanism(epsilon=epsilon, T_list=T_candidates, random_state=random_state)
    
    def run_all_conditions(self):
        """运行所有查询条件的实验"""
        all_results = []
        
        # 这里可以扩展不同的查询条件
        conditions = [
            ('Original_BUILDING', self.loader),  # 原始条件
        ]
        
        for condition_name, loader in conditions:
            condition_results = self.run_experiment_for_condition(condition_name, loader)
            if not condition_results.empty:
                all_results.append(condition_results)
        
        # 合并所有结果
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            self.analyze_cross_condition_results(final_results)
            return final_results
        else:
            print("没有获得有效结果")
            return pd.DataFrame()
    
    def analyze_cross_condition_results(self, results):
        """分析跨条件结果"""
        print(f"\n{'='*60}")
        print("跨条件结果分析")
        print(f"{'='*60}")
        
        if results.empty:
            print("没有结果可分析")
            return
        
        # 检查必要的列
        print("结果数据列:", results.columns.tolist())
        print(f"总记录数: {len(results)}")
        
        # 确保method列存在
        if 'method' not in results.columns:
            print("错误: 结果中缺少'method'列")
            return
        
        # 按条件和方法分组分析
        condition_summary = results.groupby(['condition', 'method']).agg({
            'relative_error': ['mean', 'std'],
            'kendall_tau': ['mean', 'std'],
            'jaccard': 'mean'
        }).round(4)
        
        print("各条件性能总结:")
        print(condition_summary)
        
        # 按条件分组的总体统计
        overall_summary = results.groupby('condition').agg({
            'relative_error': ['mean', 'std'],
            'kendall_tau': ['mean', 'std'],
            'jaccard': 'mean'
        }).round(4)
        
        print(f"\n各条件总体性能:")
        print(overall_summary)
        
        # 敏感度与性能关系分析
        sensitivity_correlation = results.groupby('condition').agg({
            'delta_f': 'first',
            'relative_error': 'mean'
        })
        
        print(f"\n敏感度与误差关系:")
        print(sensitivity_correlation)
        
        # 可视化结果
        self._plot_cross_condition_results(results)
    
    def _plot_cross_condition_results(self, results):
        """绘制跨条件结果图"""
        import matplotlib.pyplot as plt
        
        if results.empty or 'method' not in results.columns:
            print("没有足够的数据进行可视化")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 方法性能对比
        if 'method' in results.columns and 'relative_error' in results.columns:
            error_data = results.groupby('method')['relative_error'].agg(['mean', 'std']).reset_index()
            bars = axes[0].bar(error_data['method'], error_data['mean'], 
                             yerr=error_data['std'], capsize=5, alpha=0.7)
            axes[0].set_ylabel('Relative Error')
            axes[0].set_title('Relative Error by Method')
            axes[0].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, (_, row) in zip(bars, error_data.iterrows()):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 2. 排序质量对比
        if 'method' in results.columns and 'kendall_tau' in results.columns:
            tau_data = results.groupby('method')['kendall_tau'].agg(['mean', 'std']).reset_index()
            bars = axes[1].bar(tau_data['method'], tau_data['mean'], 
                             yerr=tau_data['std'], capsize=5, alpha=0.7)
            axes[1].set_ylabel('Kendall Tau')
            axes[1].set_title('Ranking Quality by Method')
            axes[1].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, (_, row) in zip(bars, tau_data.iterrows()):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('../results/cross_condition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n可视化结果已保存到: ../results/cross_condition_analysis.png")

def main():
    """主函数"""
    experiment = MultiConditionExperiment()
    results = experiment.run_all_conditions()
    
    if not results.empty:
        print(f"\n实验完成! 共收集 {len(results)} 条结果记录")
        print(f"方法分布: {results['method'].value_counts().to_dict()}")
        print("跨条件分析图表已保存到: ../results/cross_condition_analysis.png")
    else:
        print("\n实验完成，但没有获得有效结果")

if __name__ == "__main__":
    main()