import pandas as pd
import numpy as np
import logging
import sys
import os

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.data_loader import DataLoader
from core.naive_laplace import NaiveLaplaceMechanism
from core.r2t import R2TMechanism
from core.shifted_inverse import ShiftedInverseMechanism
from evaluation.evaluator import run_multiple_trials
from utils.config import get_db_connection_string
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, output_dir="../results"):
        self.data_loader = DataLoader(get_db_connection_string())
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_single_experiment(self, epsilon, n_trials=10):
        """运行单个epsilon值的实验"""
        logger.info(f"开始实验: ε={epsilon}, 试验次数={n_trials}")
        
        all_results = []
        
        # 初始化机制
        mechanisms = {
            'NaiveLaplace': NaiveLaplaceMechanism(self.data_loader),
            'R2T': R2TMechanism(self.data_loader),
            'ShiftedInverse': ShiftedInverseMechanism(self.data_loader)
        }
        
        for method_name, mechanism in mechanisms.items():
            logger.info(f"运行 {method_name} 机制...")
            
            try:
                # 运行多次试验
                method_results = run_multiple_trials(
                    mechanism.run_mechanism, 
                    epsilon, 
                    n_trials, 
                    method_name
                )
                
                if not method_results.empty:
                    all_results.append(method_results)
                else:
                    logger.warning(f"{method_name} 没有产生有效结果")
                    
            except Exception as e:
                logger.error(f"运行 {method_name} 失败: {e}")
                continue
        
        if not all_results:
            raise ValueError("所有方法都失败了，没有产生任何结果")
        
        # 合并结果
        final_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"实验完成，共收集 {len(final_results)} 条结果记录")
        
        return final_results
    
    def run_comprehensive_experiment(self, epsilon_list=[0.1, 0.5, 1.0, 2.0], n_trials=30):
        """运行完整的实验"""
        all_experiment_results = []
        
        for epsilon in epsilon_list:
            try:
                logger.info(f"开始 ε={epsilon} 的实验")
                results = self.run_single_experiment(epsilon, n_trials)
                all_experiment_results.append(results)
                logger.info(f"ε={epsilon} 的实验完成")
            except Exception as e:
                logger.error(f"ε={epsilon} 的实验失败: {e}")
                continue
        
        if not all_experiment_results:
            raise ValueError("所有epsilon值的实验都失败了")
        
        # 合并所有结果
        final_results = pd.concat(all_experiment_results, ignore_index=True)
        
        # 保存结果
        output_file = os.path.join(self.output_dir, "experiment_results.csv")
        final_results.to_csv(output_file, index=False)
        logger.info(f"实验结果已保存到: {output_file}")
        
        # 显示结果预览
        print("\n实验结果预览:")
        print(final_results.head())
        print(f"\n总记录数: {len(final_results)}")
        print(f"方法分布:")
        print(final_results['method'].value_counts())
        print(f"Epsilon分布:")
        print(final_results['epsilon'].value_counts().sort_index())
        
        return final_results
    
    def analyze_results(self, results):
        """分析实验结果并生成图表"""
        if results.empty:
            logger.error("没有结果数据可供分析")
            return None
            
        # 检查必要的列是否存在
        required_columns = ['jaccard', 'relative_error', 'kendall_tau', 'method', 'epsilon']
        missing_columns = [col for col in required_columns if col not in results.columns]
        if missing_columns:
            logger.error(f"结果中缺少必要的列: {missing_columns}")
            print("可用的列:", results.columns.tolist())
            return None
        
        # 计算每个方法和epsilon的平均指标
        summary = results.groupby(['method', 'epsilon']).agg({
            'jaccard': ['mean', 'std', 'count'],
            'relative_error': ['mean', 'std'],
            'kendall_tau': ['mean', 'std']
        }).round(4)
        
        print("\n实验结果总结:")
        print(summary)
        
        # 保存总结
        summary_file = os.path.join(self.output_dir, "results_summary.csv")
        summary.to_csv(summary_file)
        logger.info(f"结果总结已保存到: {summary_file}")
        
        # 生成图表
        self._plot_results(results)
        
        return summary
    
    def _plot_results(self, results):
        """生成结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Jaccard相似度
        self._plot_metric(axes[0, 0], results, 'jaccard', 'Jaccard Similarity')
        
        # 相对误差
        self._plot_metric(axes[0, 1], results, 'relative_error', 'Relative Error', log_scale=True)
        
        # Kendall Tau
        self._plot_metric(axes[1, 0], results, 'kendall_tau', 'Kendall Tau Correlation')
        
        # 方法对比箱线图
        if not results.empty:
            sns.boxplot(data=results, x='method', y='jaccard', ax=axes[1, 1])
            axes[1, 1].set_title('Jaccard Similarity Distribution by Method')
            axes[1, 1].set_ylabel('Jaccard Similarity')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "results_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"分析图表已保存到: {plot_file}")
    
    def _plot_metric(self, ax, results, metric, title, log_scale=False):
        """绘制单个指标的图表"""
        for method in results['method'].unique():
            method_data = results[results['method'] == method]
            means = method_data.groupby('epsilon')[metric].mean()
            stds = method_data.groupby('epsilon')[metric].std()
            
            ax.plot(means.index, means.values, 'o-', label=method, linewidth=2, markersize=8)
            ax.fill_between(means.index, 
                          means.values - stds.values, 
                          means.values + stds.values, 
                          alpha=0.2)
        
        ax.set_xlabel('ε (Privacy Budget)')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs Privacy Budget')
        ax.legend()
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

def main():
    """主函数"""
    runner = ExperimentRunner()
    
    try:
        # 运行实验
        logger.info("开始差分隐私机制对比实验")
        results = runner.run_comprehensive_experiment(
            epsilon_list=[0.5, 1.0, 2.0],  # 先从较大的epsilon开始测试
            n_trials=10  # 为了快速测试，先设为10次
        )
        
        # 分析结果
        summary = runner.analyze_results(results)
        
        if summary is not None:
            logger.info("实验完成！")
        else:
            logger.error("实验完成但分析失败")
            
    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()