import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class AdvancedAnalyzer:
    def __init__(self, results_file="../results/experiment_results.csv"):
        self.results = pd.read_csv(results_file)
        self.output_dir = "../results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def comprehensive_analysis(self):
        """进行全面的结果分析"""
        print("=" * 80)
        print("TPC-H Q3 差分隐私机制综合分析报告")
        print("=" * 80)
        
        # 基本统计
        self._basic_statistics()
        
        # 深入分析
        self._revenue_analysis()
        self._ranking_analysis()
        self._method_comparison()
        
        # 生成高级图表
        self._create_advanced_plots()
        
        print("\n分析完成！详细图表已保存到 results 目录")
    
    def _basic_statistics(self):
        """基本统计分析"""
        print("\n1. 基本统计信息:")
        print("-" * 40)
        
        methods = self.results['method'].unique()
        epsilons = self.results['epsilon'].unique()
        
        print(f"方法数量: {len(methods)}")
        print(f"Epsilon值: {sorted(epsilons)}")
        print(f"总试验次数: {len(self.results)}")
        print(f"每个方法-epsilon组合的试验次数: {len(self.results) // (len(methods) * len(epsilons))}")
        
        # 各方法统计
        for method in methods:
            method_data = self.results[self.results['method'] == method]
            print(f"\n{method}:")
            print(f"  相对误差均值: {method_data['relative_error'].mean():.3f} ± {method_data['relative_error'].std():.3f}")
            print(f"  Kendall Tau均值: {method_data['kendall_tau'].mean():.3f} ± {method_data['kendall_tau'].std():.3f}")
    
    def _revenue_analysis(self):
        """收入估计误差分析"""
        print("\n2. 收入估计误差分析:")
        print("-" * 40)
        
        # 按方法和epsilon分组分析
        revenue_summary = self.results.groupby(['method', 'epsilon'])['relative_error'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(4)
        
        print("收入相对误差统计:")
        print(revenue_summary)
        
        # 误差减少趋势分析
        print("\n误差随ε增加的变化趋势:")
        for method in self.results['method'].unique():
            method_data = self.results[self.results['method'] == method]
            trend = method_data.groupby('epsilon')['relative_error'].mean()
            reduction = (trend.iloc[0] - trend.iloc[-1]) / trend.iloc[0] * 100
            print(f"  {method}: 误差减少 {reduction:.1f}% (从 {trend.iloc[0]:.3f} 到 {trend.iloc[-1]:.3f})")
    
    def _ranking_analysis(self):
        """排序质量分析"""
        print("\n3. 排序质量分析:")
        print("-" * 40)
        
        # Kendall Tau分析
        tau_summary = self.results.groupby(['method', 'epsilon'])['kendall_tau'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)
        
        print("Kendall Tau排序质量统计:")
        print(tau_summary)
        
        # 统计显著性检验
        print("\n排序质量统计显著性检验 (t-test):")
        methods = self.results['method'].unique()
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                data1 = self.results[self.results['method'] == method1]['kendall_tau']
                data2 = self.results[self.results['method'] == method2]['kendall_tau']
                t_stat, p_value = stats.ttest_ind(data1, data2)
                print(f"  {method1} vs {method2}: p-value = {p_value:.4f} {'(显著)' if p_value < 0.05 else ''}")
    
    def _method_comparison(self):
        """方法对比分析"""
        print("\n4. 方法综合对比:")
        print("-" * 40)
        
        # 创建综合评分（误差越小越好，排序质量越高越好）
        self.results['error_score'] = 1 / (1 + self.results['relative_error'])  # 误差得分
        self.results['ranking_score'] = (self.results['kendall_tau'] + 1) / 2   # 排序得分归一化到[0,1]
        self.results['composite_score'] = 0.6 * self.results['error_score'] + 0.4 * self.results['ranking_score']
        
        score_summary = self.results.groupby(['method', 'epsilon'])[['error_score', 'ranking_score', 'composite_score']].mean()
        
        print("综合评分 (误差得分 + 排序得分):")
        print(score_summary.round(4))
        
        # 找出最佳方法
        best_overall = self.results.groupby('method')['composite_score'].mean().idxmax()
        best_error = self.results.groupby('method')['error_score'].mean().idxmax()
        best_ranking = self.results.groupby('method')['ranking_score'].mean().idxmax()
        
        print(f"\n最佳综合表现: {best_overall}")
        print(f"最佳收入估计: {best_error}")
        print(f"最佳排序质量: {best_ranking}")
    
    def _create_advanced_plots(self):
        """创建高级分析图表"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 相对误差趋势图
        plt.subplot(3, 3, 1)
        self._plot_error_trend()
        
        # 2. 排序质量趋势图
        plt.subplot(3, 3, 2)
        self._plot_ranking_trend()
        
        # 3. 综合评分雷达图
        plt.subplot(3, 3, 3)
        self._plot_radar_chart()
        
        # 4. 误差分布箱线图
        plt.subplot(3, 3, 4)
        self._plot_error_distribution()
        
        # 5. 排序质量分布箱线图
        plt.subplot(3, 3, 5)
        self._plot_ranking_distribution()
        
        # 6. 综合评分趋势
        plt.subplot(3, 3, 6)
        self._plot_composite_score()
        
        # 7. 方法性能热力图
        plt.subplot(3, 3, 7)
        self._plot_performance_heatmap()
        
        # 8. 误差与排序质量散点图
        plt.subplot(3, 3, 8)
        self._plot_scatter_analysis()
        
        # 9. 稳定性分析
        plt.subplot(3, 3, 9)
        self._plot_stability_analysis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comprehensive_analysis.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_error_trend(self):
        """绘制误差趋势图"""
        for method in self.results['method'].unique():
            method_data = self.results[self.results['method'] == method]
            means = method_data.groupby('epsilon')['relative_error'].mean()
            stds = method_data.groupby('epsilon')['relative_error'].std()
            
            plt.plot(means.index, means.values, 'o-', label=method, linewidth=2, markersize=8)
            plt.fill_between(means.index, means - stds, means + stds, alpha=0.2)
        
        plt.xlabel('ε (Privacy Budget)')
        plt.ylabel('Relative Error')
        plt.title('Revenue Estimation Error vs Privacy Budget')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # plt.yscale('log')
    
    def _plot_ranking_trend(self):
        """绘制排序质量趋势图"""
        for method in self.results['method'].unique():
            method_data = self.results[self.results['method'] == method]
            means = method_data.groupby('epsilon')['kendall_tau'].mean()
            stds = method_data.groupby('epsilon')['kendall_tau'].std()
            
            plt.plot(means.index, means.values, 'o-', label=method, linewidth=2, markersize=8)
            plt.fill_between(means.index, means - stds, means + stds, alpha=0.2)
        
        plt.xlabel('ε (Privacy Budget)')
        plt.ylabel('Kendall Tau')
        plt.title('Ranking Quality vs Privacy Budget')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_radar_chart(self):
        """绘制雷达图"""
        # 简化版雷达图 - 使用条形图替代
        scores = self.results.groupby('method')[['error_score', 'ranking_score']].mean()
        
        x = np.arange(len(scores))
        width = 0.35
        
        plt.bar(x - width/2, scores['error_score'], width, label='Error Score', alpha=0.8)
        plt.bar(x + width/2, scores['ranking_score'], width, label='Ranking Score', alpha=0.8)
        
        plt.xlabel('Method')
        plt.ylabel('Score')
        plt.title('Method Performance Scores')
        plt.xticks(x, scores.index)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_error_distribution(self):
        """绘制误差分布箱线图"""
        sns.boxplot(data=self.results, x='method', y='relative_error')
        plt.title('Revenue Error Distribution by Method')
        plt.xticks(rotation=45)
        plt.yscale('log')
    
    def _plot_ranking_distribution(self):
        """绘制排序质量分布箱线图"""
        sns.boxplot(data=self.results, x='method', y='kendall_tau')
        plt.title('Ranking Quality Distribution by Method')
        plt.xticks(rotation=45)
    
    def _plot_composite_score(self):
        """绘制综合评分趋势"""
        for method in self.results['method'].unique():
            method_data = self.results[self.results['method'] == method]
            means = method_data.groupby('epsilon')['composite_score'].mean()
            plt.plot(means.index, means.values, 'o-', label=method, linewidth=2, markersize=8)
        
        plt.xlabel('ε (Privacy Budget)')
        plt.ylabel('Composite Score')
        plt.title('Overall Performance vs Privacy Budget')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_performance_heatmap(self):
        """绘制性能热力图"""
        pivot_table = self.results.pivot_table(
            values='composite_score', 
            index='method', 
            columns='epsilon', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', cbar_kws={'label': 'Composite Score'})
        plt.title('Performance Heatmap\n(Method vs Privacy Budget)')
    
    def _plot_scatter_analysis(self):
        """绘制误差与排序质量散点图"""
        for method in self.results['method'].unique():
            method_data = self.results[self.results['method'] == method]
            plt.scatter(method_data['relative_error'], method_data['kendall_tau'], 
                       label=method, alpha=0.6, s=50)
        
        plt.xlabel('Relative Error')
        plt.ylabel('Kendall Tau')
        plt.title('Error vs Ranking Quality Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_stability_analysis(self):
        """绘制稳定性分析图"""
        stability = self.results.groupby('method')['relative_error'].std()
        plt.bar(stability.index, stability.values, alpha=0.7)
        plt.xlabel('Method')
        plt.ylabel('Standard Deviation of Error')
        plt.title('Method Stability (Lower is Better)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

def main():
    """主分析函数"""
    analyzer = AdvancedAnalyzer()
    analyzer.comprehensive_analysis()

if __name__ == "__main__":
    main()