# src/data_driven_parameter_selector.py
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataDrivenParameterSelector:
    """完全数据驱动的参数选择器"""
    
    def __init__(self, data_analyzer):
        self.data_analyzer = data_analyzer
        self.parameter_cache = defaultdict(dict)
        self.global_features = None
    
    def initialize_global_analysis(self):
        """初始化全局分析"""
        logger.info("初始化全局数据分析...")
        self.global_features = self.data_analyzer.analyze_tpch_global_features()
        logger.info("全局数据分析完成")
    
    def suggest_parameters_for_query(self, market_segment, date, mechanism_type):
        """为特定查询推荐参数"""
        # 首先分析该查询模式的数据特征
        query_stats = self.data_analyzer.analyze_specific_query_pattern(market_segment, date)
        
        if query_stats is None:
            logger.warning(f"无法分析查询模式 {market_segment} {date}，使用保守参数")
            return self._get_conservative_parameters(mechanism_type)
        
        # 基于实际数据特征推荐参数
        if mechanism_type == 'R2T':
            return self._suggest_r2t_parameters(query_stats)
        elif mechanism_type == 'ShiftedInverse':
            return self._suggest_shifted_inverse_parameters(query_stats)
        else:
            return self._get_conservative_parameters(mechanism_type)
    
    def _suggest_r2t_parameters(self, query_stats):
        """基于数据特征推荐R2T参数"""
        complexity = query_stats.get('complexity_score', 0.5)
        customer_count = query_stats.get('customer_count', 100)
        
        # 基于百分位数选择T值
        item_quantiles = [
            query_stats.get('p10_items', 1),
            query_stats.get('p25_items', 2), 
            query_stats.get('p50_items', 3),
            query_stats.get('p75_items', 5),
            query_stats.get('p90_items', 8)
        ]
        
        # 根据复杂度调整策略
        if complexity < 0.3:
            # 低复杂度：更激进的截断
            selected_quantiles = [0, 1]  # p10, p25
        elif complexity < 0.6:
            # 中等复杂度：平衡策略
            selected_quantiles = [1, 2, 3]  # p25, p50, p75
        else:
            # 高复杂度：保守截断
            selected_quantiles = [2, 3, 4]  # p50, p75, p90
        
        T_candidates = [max(1, int(item_quantiles[i])) for i in selected_quantiles]
        
        # 去除重复并排序
        T_candidates = sorted(list(set(T_candidates)))
        
        # 确保不超过最大值
        max_items = query_stats.get('max_items_per_customer', 20)
        T_candidates = [min(t, max_items) for t in T_candidates]
        
        logger.info(f"R2T参数建议: T_candidates = {T_candidates} "
                   f"(复杂度: {complexity:.2f}, 客户数: {customer_count})")
        
        return T_candidates
    
    def _suggest_shifted_inverse_parameters(self, query_stats):
        """基于数据特征推荐Shifted Inverse参数"""
        complexity = query_stats.get('complexity_score', 0.5)
        max_contribution = query_stats.get('max_contribution_per_customer', 100000)
        
        # 基于贡献值百分位数选择T值
        contribution_quantiles = [
            query_stats.get('p10_contribution', 10000),
            query_stats.get('p25_contribution', 30000),
            query_stats.get('p50_contribution', 60000),
            query_stats.get('p75_contribution', 120000),
            query_stats.get('p90_contribution', 200000)
        ]
        
        # 根据复杂度调整策略
        if complexity < 0.4:
            # 低复杂度：中等截断
            selected_quantiles = [1, 2]  # p25, p50
        elif complexity < 0.7:
            # 中等复杂度：平衡策略
            selected_quantiles = [2, 3]  # p50, p75
        else:
            # 高复杂度：保守截断
            selected_quantiles = [3, 4]  # p75, p90
        
        T_candidates = [max(1000, int(contribution_quantiles[i])) for i in selected_quantiles]
        
        # 基于数据丰富度进一步调整
        richness = query_stats.get('data_richness', 0.5)
        if richness > 0.7:
            # 数据丰富，可以更激进
            T_candidates = [int(t * 0.8) for t in T_candidates]
        elif richness < 0.3:
            # 数据稀疏，需要更保守
            T_candidates = [int(t * 1.2) for t in T_candidates]
        
        # 确保不超过最大贡献的80%
        T_candidates = [min(t, int(max_contribution * 0.8)) for t in T_candidates]
        T_candidates = sorted(list(set(T_candidates)))
        
        logger.info(f"Shifted Inverse参数建议: T_candidates = {T_candidates} "
                   f"(复杂度: {complexity:.2f}, 丰富度: {richness:.2f})")
        
        return T_candidates
    
    def _get_conservative_parameters(self, mechanism_type):
        """获取保守的参数备选"""
        if mechanism_type == 'R2T':
            return [1, 3, 5, 10]
        elif mechanism_type == 'ShiftedInverse':
            return [50000, 100000, 200000]
        else:
            return []
    
    def cache_parameters(self, segment, date, mechanism_type, parameters, performance_metrics=None):
        """缓存参数及性能指标"""
        key = f"{segment}_{date}"
        self.parameter_cache[key][mechanism_type] = {
            'parameters': parameters,
            'performance': performance_metrics,
            'timestamp': pd.Timestamp.now(),
            'usage_count': 0
        }
    
    def get_cached_parameters(self, segment, date, mechanism_type, max_age_hours=24):
        """获取缓存的参数，包含使用统计"""
        key = f"{segment}_{date}"
        if key in self.parameter_cache:
            cache_entry = self.parameter_cache[key].get(mechanism_type)
            if cache_entry:
                age = (pd.Timestamp.now() - cache_entry['timestamp']).total_seconds() / 3600
                if age < max_age_hours:
                    cache_entry['usage_count'] += 1
                    logger.info(f"使用缓存参数 [{segment}-{date}], 使用次数: {cache_entry['usage_count']}")
                    return cache_entry['parameters']
        return None
    
    def analyze_parameter_performance(self):
        """分析参数选择性能"""
        performance_data = []
        
        for key, mechanisms in self.parameter_cache.items():
            segment, date = key.split('_', 1)
            for mech_type, cache_entry in mechanisms.items():
                if cache_entry.get('performance'):
                    performance_data.append({
                        'segment': segment,
                        'date': date,
                        'mechanism': mech_type,
                        'parameters': cache_entry['parameters'],
                        'performance': cache_entry['performance'],
                        'usage_count': cache_entry['usage_count']
                    })
        
        return pd.DataFrame(performance_data)