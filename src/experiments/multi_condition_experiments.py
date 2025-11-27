import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.data_loader import DataLoader
from utils.config import get_db_connection_string
from evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiConditionExperiment:
    """多查询条件实验 - 完整版本"""
    
    def __init__(self):
        self.loader = DataLoader(get_db_connection_string())
    
    def run_experiment_for_condition(self, market_segment, date='1995-03-15'):
        """为特定市场细分和日期运行实验"""
        condition_name = f"{market_segment}_{date.replace('-', '')}"
        
        print(f"\n{'='*60}")
        print(f"运行实验: {condition_name}")
        print(f"{'='*60}")
        
        try:
            # 使用指定的市场细分和日期获取数据
            ground_truth = self.loader.get_ground_truth(market_segment, date)
            contributions = self.loader.get_customer_contributions(market_segment, date)
            
            if ground_truth.empty or len(ground_truth) < 5:  # 至少需要5个结果才有意义
                print(f"  ⚠️ {condition_name} 数据不足({len(ground_truth)}条)，跳过")
                return pd.DataFrame()
            
            print(f"数据特征:")
            print(f"  • 客户数量: {contributions['c_custkey'].nunique()}")
            print(f"  • 订单项数量: {len(contributions)}")
            print(f"  • Top-10订单数量: {len(ground_truth)}")
            print(f"  • Top-10订单总收入: {ground_truth['revenue'].sum():.2f}")
            
            # 分析敏感度
            customer_totals = contributions.groupby('c_custkey')['contribution'].sum()
            delta_f = customer_totals.max()
            sensitivity_ratio = delta_f / ground_truth['revenue'].mean() if ground_truth['revenue'].mean() > 0 else 0
            
            print(f"  • 全局敏感度 Δf: {delta_f:.2f}")
            print(f"  • Δf/平均收入: {sensitivity_ratio:.2f}")
            
            results = []
            
            # 测试各种机制
            mechanisms = [
                ('NaiveLaplace', self._run_naive),
                ('Adaptive_R2T', self._run_adaptive_r2t),
                ('Adaptive_SI', self._run_adaptive_si)
            ]
            
            for method_name, mechanism_func in mechanisms:
                try:
                    # 传递市场细分和日期参数
                    result = mechanism_func(market_segment, date, epsilon=1.0, random_state=42)
                    if result is None or result.empty:
                        continue
                        
                    evaluator = Evaluator(ground_truth)
                    metrics = evaluator.evaluate_all(result, method_name)
                    
                    # 确保包含所有必要字段
                    metrics['method'] = method_name
                    metrics['condition'] = condition_name
                    metrics['market_segment'] = market_segment
                    metrics['date'] = date
                    metrics['delta_f'] = delta_f
                    metrics['sensitivity_ratio'] = sensitivity_ratio
                    metrics['customer_count'] = contributions['c_custkey'].nunique()
                    metrics['item_count'] = len(contributions)
                    metrics['topk_count'] = len(ground_truth)
                    
                    results.append(metrics)
                    
                    print(f"  {method_name}: 误差={metrics['relative_error']:.3f}, Tau={metrics['kendall_tau']:.3f}")
                    
                except Exception as e:
                    print(f"  {method_name} 失败: {e}")
                    continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"  {condition_name} 数据加载失败: {e}")
            return pd.DataFrame()
    
    def _run_naive(self, market_segment, date, epsilon, random_state):
        """运行朴素拉普拉斯机制"""
        from core.naive_laplace import NaiveLaplaceMechanism
        mechanism = NaiveLaplaceMechanism(self.loader)
        return mechanism.run_mechanism(
            epsilon=epsilon, 
            market_segment=market_segment,
            date=date,
            random_state=random_state
        )
    
    def _run_adaptive_r2t(self, market_segment, date, epsilon, random_state):
        """运行自适应R2T"""
        from core.r2t import R2TMechanism
        
        try:
            # 使用指定参数获取贡献数据
            contributions = self.loader.get_customer_contributions(market_segment, date)
            
            # 基于分位数选择T值
            customer_stats = contributions.groupby('c_custkey').agg({
                'contribution': ['sum', 'count']
            })
            item_counts = customer_stats[('contribution', 'count')]
            
            quantiles = item_counts.quantile([0.25, 0.5, 0.75])
            T_candidates = [max(1, int(q)) for q in quantiles]
            
            mechanism = R2TMechanism(self.loader)
            return mechanism.run_mechanism(
                epsilon=epsilon, 
                T_list=T_candidates,
                market_segment=market_segment,
                date=date,
                random_state=random_state
            )
        except Exception as e:
            print(f"  R2T机制失败: {e}")
            return None
    
    def _run_adaptive_si(self, market_segment, date, epsilon, random_state):
        """运行自适应Shifted Inverse"""
        from core.shifted_inverse import ShiftedInverseMechanism
        
        try:
            # 使用指定参数获取贡献数据
            contributions = self.loader.get_customer_contributions(market_segment, date)
            
            # 基于分位数选择T值
            customer_stats = contributions.groupby('c_custkey').agg({
                'contribution': ['sum', 'count']
            })
            customer_totals = customer_stats[('contribution', 'sum')]
            
            quantiles = customer_totals.quantile([0.25, 0.5, 0.75])
            T_candidates = [max(1000, int(q)) for q in quantiles]
            
            mechanism = ShiftedInverseMechanism(self.loader)
            return mechanism.run_mechanism(
                epsilon=epsilon, 
                T_list=T_candidates,
                market_segment=market_segment,
                date=date,
                random_state=random_state
            )
        except Exception as e:
            print(f"  Shifted Inverse机制失败: {e}")
            return None
    
    def run_complete_experiment(self):
        """运行完整的多条件实验"""
        all_results = []
        
        # 完整的测试条件：5个市场细分 × 5个日期
        market_segments = ['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE']
        dates = ['1995-03-01', '1995-03-08', '1995-03-15', '1995-03-22', '1995-03-29']
        
        print("开始完整多条件实验...")
        print(f"测试市场细分: {market_segments}")
        print(f"测试日期: {dates}")
        print(f"总测试条件: {len(market_segments) * len(dates)}")
        print(f"{'='*60}")
        
        total_conditions = len(market_segments) * len(dates)
        completed_conditions = 0
        
        for segment in market_segments:
            for date in dates:
                try:
                    print(f"\n进度: {completed_conditions + 1}/{total_conditions}")
                    condition_results = self.run_experiment_for_condition(segment, date)
                    if not condition_results.empty:
                        all_results.append(condition_results)
                        completed_conditions += 1
                        print(f"✅ 完成 {segment} {date} 的实验")
                    else:
                        print(f"⚠️ {segment} {date} 无有效结果")
                except Exception as e:
                    print(f"❌ {segment} {date} 实验失败: {e}")
                    continue
        
        # 合并所有结果
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            print(f"\n{'='*60}")
            print("实验完成总结")
            print(f"{'='*60}")
            print(f"总测试条件: {total_conditions}")
            print(f"成功完成: {completed_conditions}")
            print(f"成功率: {completed_conditions/total_conditions*100:.1f}%")
            
            self.analyze_comprehensive_results(final_results)
            return final_results
        else:
            print("所有实验均无有效结果")
            return pd.DataFrame()
    
    def analyze_comprehensive_results(self, results):
        """分析完整实验结果"""
        print(f"\n{'='*60}")
        print("完整多条件实验结果分析")
        print(f"{'='*60}")
        
        if results.empty:
            print("没有结果可分析")
            return
        
        print(f"总记录数: {len(results)}")
        print(f"成功测试条件数: {results['condition'].nunique()}")
        print(f"涉及市场细分: {sorted(results['market_segment'].unique())}")
        print(f"涉及日期范围: {sorted(results['date'].unique())}")
        print(f"方法分布: {results['method'].value_counts().to_dict()}")
        
        # 详细性能分析
        print(f"\n{'='*40}")
        print("详细性能分析")
        print(f"{'='*40}")
        
        # 按方法分组的总体性能
        overall_performance = results.groupby('method').agg({
            'relative_error': ['mean', 'std', 'min', 'max'],
            'kendall_tau': ['mean', 'std', 'min', 'max'],
            'jaccard': 'mean'
        }).round(4)
        
        print("各方法总体性能:")
        print(overall_performance)
        
        # 按市场细分分析
        segment_analysis = results.groupby(['market_segment', 'method']).agg({
            'relative_error': 'mean',
            'kendall_tau': 'mean',
            'delta_f': 'first',
            'sensitivity_ratio': 'first',
            'customer_count': 'first'
        }).round(4)
        
        print(f"\n各市场细分详细性能:")
        print(segment_analysis)
        
        # 敏感度与性能相关性分析
        print(f"\n{'='*40}")
        print("敏感度与性能相关性分析")
        print(f"{'='*40}")
        
        correlation_data = results[['sensitivity_ratio', 'relative_error', 'kendall_tau', 'customer_count']].corr()
        print("特征相关性矩阵:")
        print(correlation_data.round(3))
        
        # 性能稳定性分析
        stability_analysis = results.groupby('method').agg({
            'relative_error': 'std',
            'kendall_tau': 'std'
        }).round(4)
        
        print(f"\n方法性能稳定性(标准差):")
        print(stability_analysis)
        
        # 保存详细结果
        output_dir = '../results'
        os.makedirs(output_dir, exist_ok=True)
        
        results.to_csv(f'{output_dir}/comprehensive_results.csv', index=False)
        overall_performance.to_csv(f'{output_dir}/overall_performance.csv')
        segment_analysis.to_csv(f'{output_dir}/segment_analysis.csv')
        
        # 生成综合可视化
        self._create_comprehensive_visualizations(results)
        
        print(f"\n详细结果已保存到 {output_dir}/ 目录")
    
    def _create_comprehensive_visualizations(self, results):
        """创建综合可视化图表"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if results.empty:
            print("没有足够的数据进行可视化")
            return
        
        # 创建2x3的子图布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Multi-Condition Analysis', fontsize=16, fontweight='bold')
        
        # 1. 各方法性能对比热力图
        if all(col in results.columns for col in ['market_segment', 'method', 'relative_error']):
            performance_pivot = results.pivot_table(
                values='relative_error', 
                index='market_segment', 
                columns='method', 
                aggfunc='mean'
            )
            sns.heatmap(performance_pivot, annot=True, cmap='YlOrRd_r', 
                       cbar_kws={'label': 'Relative Error'}, ax=axes[0,0])
            axes[0,0].set_title('A. Relative Error by Segment and Method')
        
        # 2. 排序质量热力图
        if all(col in results.columns for col in ['market_segment', 'method', 'kendall_tau']):
            tau_pivot = results.pivot_table(
                values='kendall_tau', 
                index='market_segment', 
                columns='method', 
                aggfunc='mean'
            )
            sns.heatmap(tau_pivot, annot=True, cmap='RdYlBu', center=0,
                       cbar_kws={'label': 'Kendall Tau'}, ax=axes[0,1])
            axes[0,1].set_title('B. Ranking Quality by Segment and Method')
        
        # 3. 敏感度与误差关系
        if all(col in results.columns for col in ['sensitivity_ratio', 'relative_error', 'method']):
            for method in results['method'].unique():
                method_data = results[results['method'] == method]
                axes[0,2].scatter(method_data['sensitivity_ratio'], 
                                method_data['relative_error'], 
                                label=method, alpha=0.6, s=60)
            axes[0,2].set_xlabel('Sensitivity Ratio (Δf/avg_revenue)')
            axes[0,2].set_ylabel('Relative Error')
            axes[0,2].set_title('C. Sensitivity vs Revenue Error')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. 客户数量与性能关系
        if all(col in results.columns for col in ['customer_count', 'relative_error', 'method']):
            for method in results['method'].unique():
                method_data = results[results['method'] == method]
                axes[1,0].scatter(method_data['customer_count'], 
                                method_data['relative_error'], 
                                label=method, alpha=0.6, s=60)
            axes[1,0].set_xlabel('Customer Count')
            axes[1,0].set_ylabel('Relative Error')
            axes[1,0].set_title('D. Customer Count vs Performance')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. 方法性能分布箱线图
        if 'method' in results.columns and 'relative_error' in results.columns:
            sns.boxplot(data=results, x='method', y='relative_error', ax=axes[1,1])
            axes[1,1].set_title('E. Revenue Error Distribution by Method')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. 排序质量分布箱线图
        if 'method' in results.columns and 'kendall_tau' in results.columns:
            sns.boxplot(data=results, x='method', y='kendall_tau', ax=axes[1,2])
            axes[1,2].set_title('F. Ranking Quality Distribution by Method')
            axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"综合可视化结果已保存到: ../results/comprehensive_analysis.png")

def main():
    """主函数"""
    experiment = MultiConditionExperiment()
    results = experiment.run_complete_experiment()
    
    if not results.empty:
        print(f"\n🎉 完整实验完成!")
        print(f"共成功测试 {results['condition'].nunique()} 个条件")
        print(f"收集 {len(results)} 条结果记录")
        print(f"涉及所有5个细分市场")
        print(f"详细结果和图表已保存到 ../results/ 目录")
    else:
        print("\n实验完成，但没有获得有效结果")

if __name__ == "__main__":
    main()