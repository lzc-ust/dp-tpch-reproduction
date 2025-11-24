# src/experiments/data_driven_experiment.py
import pandas as pd
import numpy as np
import logging
import sys
import os

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.data_loader import DataLoader
from utils.data_analyzer import DataAnalyzer
from utils.data_driven_parameter_selector import DataDrivenParameterSelector
from utils.config import get_db_connection_string
from evaluation.evaluator import Evaluator

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDrivenExperiment:
    """完全数据驱动的实验系统"""
    
    def __init__(self):
        self.loader = DataLoader(get_db_connection_string())
        self.data_analyzer = DataAnalyzer(get_db_connection_string())
        self.parameter_selector = DataDrivenParameterSelector(self.data_analyzer)
        
        # 初始化全局分析 - 改为延迟初始化
        try:
            self.parameter_selector.initialize_global_analysis()
        except Exception as e:
            logger.warning(f"全局分析初始化失败，将使用动态分析: {e}")
    
    def run_data_driven_analysis(self):
        """运行数据驱动的全面分析"""
        print("=" * 80)
        print("数据驱动的TPC-H差分隐私分析")
        print("=" * 80)
        
        # 分析所有市场细分和日期的组合
        segments = self.loader.market_segments
        dates = self.loader.generate_tpch_dates()
        
        all_results = []
        query_patterns_analyzed = 0
        
        # 限制测试范围以避免长时间运行
        test_segments = segments[:2]  # 只测试前2个细分市场
        test_dates = dates[:2]       # 只测试前2个日期
        
        for segment in test_segments:
            for date in test_dates:
                try:
                    logger.info(f"分析 {segment} {date}...")
                    
                    # 为每个查询模式运行实验
                    results = self._run_query_pattern_experiment(segment, date)
                    if results is not None and not results.empty:
                        all_results.append(results)
                        query_patterns_analyzed += 1
                        logger.info(f"完成 {segment} {date} 的分析")
                        
                except Exception as e:
                    logger.error(f"分析 {segment} {date} 失败: {e}")
                    continue
        
        # 合并所有结果
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            self._analyze_data_driven_results(final_results)
            
            print(f"\n✅ 分析完成! 共分析了 {query_patterns_analyzed} 个查询模式")
            return final_results
        else:
            print("❌ 没有获得有效结果")
            return pd.DataFrame()
    
    def _run_query_pattern_experiment(self, segment, date, epsilon=1.0, n_trials=2):
        """运行特定查询模式的实验 - 修复method列问题"""
        results = []
        
        # 获取黄金标准
        ground_truth = self.loader.get_ground_truth(segment, date)
        
        # 测试各种机制
        mechanisms = [
            ('NaiveLaplace', self._run_naive_laplace),
            ('R2T_DataDriven', self._run_data_driven_r2t),
            ('SI_DataDriven', self._run_data_driven_si)
        ]
        
        for method_name, mechanism_func in mechanisms:
            logger.info(f"运行 {method_name} 机制...")
            method_results = []
            
            for trial in range(n_trials):
                try:
                    result = mechanism_func(segment, date, epsilon, trial)
                    
                    # 评估结果
                    evaluator = Evaluator(ground_truth)
                    metrics = evaluator.evaluate_all(result, method_name)
                    
                    # 确保包含所有必要字段
                    metrics['method'] = method_name  # 确保method列存在
                    metrics['segment'] = segment
                    metrics['date'] = date
                    metrics['trial'] = trial
                    metrics['epsilon'] = epsilon
                    
                    # 添加查询模式信息
                    query_stats = self.data_analyzer.analyze_specific_query_pattern(segment, date)
                    if query_stats is not None:
                        metrics.update({
                            'customer_count': query_stats.get('customer_count'),
                            'max_contribution': query_stats.get('max_contribution_per_customer'),
                            'complexity': query_stats.get('complexity_score', 0.5),
                            'data_richness': query_stats.get('data_richness', 0.5)
                        })
                    
                    method_results.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"{method_name} 试验 {trial} 失败: {e}")
                    continue
            
            if method_results:
                results.extend(method_results)
                logger.info(f"{method_name} 完成 {len(method_results)} 次试验")
        
        if results:
            return pd.DataFrame(results)
        else:
            return None
    
    def _run_naive_laplace(self, segment, date, epsilon, random_state):
        """运行朴素拉普拉斯机制 - 修复参数传递"""
        from core.naive_laplace import NaiveLaplaceMechanism
        
        # 直接使用主loader，但传递细分市场和日期参数
        mechanism = NaiveLaplaceMechanism(self.loader)
        return mechanism.run_mechanism(
            epsilon=epsilon, 
            market_segment=segment, 
            date=date, 
            random_state=random_state
        )
    
    def _run_data_driven_r2t(self, segment, date, epsilon, random_state):
        """运行数据驱动的R2T - 修复参数传递"""
        from core.r2t import R2TMechanism
        
        # 获取数据驱动的参数
        T_candidates = self.parameter_selector.suggest_parameters_for_query(segment, date, 'R2T')
        
        # 直接使用主loader，传递细分市场和日期参数
        mechanism = R2TMechanism(self.loader)
        result = mechanism.run_mechanism(
            epsilon=epsilon, 
            T_list=T_candidates, 
            market_segment=segment, 
            date=date, 
            random_state=random_state
        )
        
        # 缓存参数
        self.parameter_selector.cache_parameters(segment, date, 'R2T', T_candidates)
        
        return result
    
    def _run_data_driven_si(self, segment, date, epsilon, random_state):
        """运行数据驱动的Shifted Inverse - 修复参数传递"""
        from core.shifted_inverse import ShiftedInverseMechanism
        
        # 获取数据驱动的参数
        T_candidates = self.parameter_selector.suggest_parameters_for_query(segment, date, 'ShiftedInverse')
        
        # 直接使用主loader，传递细分市场和日期参数
        mechanism = ShiftedInverseMechanism(self.loader)
        result = mechanism.run_mechanism(
            epsilon=epsilon, 
            T_list=T_candidates, 
            market_segment=segment, 
            date=date, 
            random_state=random_state
        )
        
        # 缓存参数
        self.parameter_selector.cache_parameters(segment, date, 'ShiftedInverse', T_candidates)
        
        return result
    
    def _analyze_data_driven_results(self, results):
        """分析数据驱动结果"""
        print("\n" + "=" * 80)
        print("数据驱动实验结果分析")
        print("=" + "-" * 79)
        
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
        
        # 按细分市场和方法分析
        performance_by_segment = results.groupby(['segment', 'method']).agg({
            'relative_error': ['mean', 'std', 'count'],
            'kendall_tau': ['mean', 'std'],
            'jaccard': 'mean',
            'complexity': 'mean'
        }).round(4)
        
        print("各细分市场性能:")
        print(performance_by_segment)
        
        # 数据特征与性能关系分析
        self._analyze_feature_performance_correlation(results)
        
        # 可视化结果
        self._create_data_driven_visualizations(results)
    
    def _analyze_feature_performance_correlation(self, results):
        """分析数据特征与性能的相关性"""
        print("\n数据特征与性能相关性分析:")
        
        # 检查是否有数值列
        numeric_columns = []
        for col in ['relative_error', 'kendall_tau', 'customer_count', 'max_contribution', 'complexity', 'data_richness']:
            if col in results.columns:
                numeric_columns.append(col)
        
        if len(numeric_columns) < 2:
            print("数值列不足，无法计算相关性")
            return
        
        # 计算相关系数
        correlation_matrix = results[numeric_columns].corr()
        
        print("特征与性能相关系数:")
        if 'relative_error' in correlation_matrix.columns:
            print(correlation_matrix[['relative_error', 'kendall_tau']].round(3))
        
        # 找出对性能影响最大的特征
        if 'relative_error' in correlation_matrix.columns:
            error_correlations = correlation_matrix['relative_error'].abs().sort_values(ascending=False)
            # 跳过自身相关性
            if len(error_correlations) > 1:
                top_feature = error_correlations.index[1]
                print(f"\n对误差影响最大的特征: {top_feature} (r={error_correlations.iloc[1]:.3f})")
    
    def _create_data_driven_visualizations(self, results):
        """创建数据驱动的可视化"""
        if results.empty:
            print("没有结果可可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        plot_count = 0
        
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
            plot_count += 1
        
        # 2. 细分市场性能热力图
        if all(col in results.columns for col in ['segment', 'method', 'relative_error']):
            try:
                performance_pivot = results.pivot_table(
                    values='relative_error', 
                    index='segment', 
                    columns='method', 
                    aggfunc='mean'
                )
                sns.heatmap(performance_pivot, annot=True, cmap='YlOrRd_r', 
                           cbar_kws={'label': 'Relative Error'}, ax=axes[1])
                axes[1].set_title('Performance Heatmap by Segment and Method')
                plot_count += 1
            except Exception as e:
                logger.warning(f"热力图创建失败: {e}")
        
        # 3. 复杂度 vs 性能
        if all(col in results.columns for col in ['complexity', 'relative_error', 'method']):
            for method in results['method'].unique():
                method_data = results[results['method'] == method]
                axes[2].scatter(method_data['complexity'], method_data['relative_error'], 
                               label=method, alpha=0.6, s=60)
            axes[2].set_xlabel('Data Complexity')
            axes[2].set_ylabel('Relative Error')
            axes[2].set_title('Data Complexity vs Performance by Method')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            plot_count += 1
        
        # 4. 客户数量 vs 性能
        if all(col in results.columns for col in ['customer_count', 'relative_error', 'method']):
            for method in results['method'].unique():
                method_data = results[results['method'] == method]
                axes[3].scatter(method_data['customer_count'], method_data['relative_error'], 
                               label=method, alpha=0.6, s=60)
            axes[3].set_xlabel('Customer Count')
            axes[3].set_ylabel('Relative Error')
            axes[3].set_title('Customer Count vs Performance by Method')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            plot_count += 1
        
        # 隐藏未使用的子图
        for i in range(plot_count, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('../results/data_driven_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n可视化结果已保存到: ../results/data_driven_analysis.png")

def main():
    """主函数"""
    try:
        experiment = DataDrivenExperiment()
        results = experiment.run_data_driven_analysis()
        
        if not results.empty:
            print(f"\n实验完成! 分析了 {results['segment'].nunique()} 个细分市场")
            print(f"收集了 {len(results)} 条实验结果")
            print(f"方法分布: {results['method'].value_counts().to_dict()}")
        else:
            print("\n实验完成，但没有获得有效结果")
            
    except Exception as e:
        print(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()