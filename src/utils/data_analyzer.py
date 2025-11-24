# src/data_analyzer.py (修复版本)
import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """完全数据驱动的特征分析器"""
    
    def __init__(self, db_connection_string):
        self.engine = create_engine(db_connection_string)
    
    def analyze_tpch_global_features(self):
        """分析TPC-H全局数据特征"""
        logger.info("开始分析TPC-H全局数据特征...")
        
        features = {}
        
        # 1. 分析各市场细分的总体特征
        segment_analysis = self._analyze_market_segments()
        features['segment_analysis'] = segment_analysis
        
        # 2. 分析时间分布特征
        date_analysis = self._analyze_date_distribution()
        features['date_analysis'] = date_analysis
        
        # 3. 分析客户行为模式
        customer_analysis = self._analyze_customer_patterns()
        features['customer_analysis'] = customer_analysis
        
        logger.info("TPC-H全局特征分析完成")
        return features
    
    def _analyze_market_segments(self):
        """分析各市场细分的统计特征 - 修复MySQL语法"""
        query = text("""
        SELECT 
            c_mktsegment as segment,
            COUNT(DISTINCT c_custkey) as customer_count,
            COUNT(DISTINCT o_orderkey) as order_count,
            COUNT(*) as lineitem_count,
            AVG(l_extendedprice * (1 - l_discount)) as avg_contribution,
            MAX(l_extendedprice * (1 - l_discount)) as max_contribution,
            SUM(l_extendedprice * (1 - l_discount)) as total_contribution,
            STDDEV(l_extendedprice * (1 - l_discount)) as std_contribution
        FROM 
            customer 
            JOIN orders ON c_custkey = o_custkey
            JOIN lineitem ON o_orderkey = l_orderkey
        GROUP BY 
            c_mktsegment
        ORDER BY 
            total_contribution DESC
        """)
        
        segment_stats = pd.read_sql(query, self.engine)
        
        # 在Python中计算中位数和其他分位数
        segments = segment_stats['segment'].unique()
        enhanced_stats = []
        
        for segment in segments:
            # 获取该细分的详细数据计算分位数
            detail_query = text(f"""
            SELECT l_extendedprice * (1 - l_discount) as contribution
            FROM customer 
            JOIN orders ON c_custkey = o_custkey
            JOIN lineitem ON o_orderkey = l_orderkey
            WHERE c_mktsegment = '{segment}'
            """)
            
            detail_data = pd.read_sql(detail_query, self.engine)
            
            segment_row = segment_stats[segment_stats['segment'] == segment].iloc[0].copy()
            segment_row['median_contribution'] = detail_data['contribution'].median()
            segment_row['p25_contribution'] = detail_data['contribution'].quantile(0.25)
            segment_row['p75_contribution'] = detail_data['contribution'].quantile(0.75)
            
            enhanced_stats.append(segment_row)
        
        segment_stats = pd.DataFrame(enhanced_stats)
        
        # 计算额外指标
        segment_stats['contribution_per_customer'] = segment_stats['total_contribution'] / segment_stats['customer_count']
        segment_stats['items_per_customer'] = segment_stats['lineitem_count'] / segment_stats['customer_count']
        segment_stats['cv_contribution'] = segment_stats['std_contribution'] / segment_stats['avg_contribution']
        
        logger.info(f"分析了 {len(segment_stats)} 个市场细分")
        return segment_stats
    
    def _analyze_date_distribution(self):
        """分析时间分布特征"""
        query = text("""
        SELECT 
            EXTRACT(YEAR FROM o_orderdate) as order_year,
            EXTRACT(MONTH FROM o_orderdate) as order_month,
            COUNT(DISTINCT o_orderkey) as order_count,
            AVG(l_extendedprice * (1 - l_discount)) as avg_contribution,
            SUM(l_extendedprice * (1 - l_discount)) as total_contribution
        FROM 
            orders 
            JOIN lineitem ON o_orderkey = l_orderkey
        GROUP BY 
            EXTRACT(YEAR FROM o_orderdate), EXTRACT(MONTH FROM o_orderdate)
        ORDER BY 
            order_year, order_month
        """)
        
        date_stats = pd.read_sql(query, self.engine)
        logger.info(f"分析了 {len(date_stats)} 个时间区间的数据分布")
        return date_stats
    
    def _analyze_customer_patterns(self):
        """分析客户行为模式 - 修复MySQL语法"""
        # 首先获取基础统计
        query = text("""
        WITH customer_stats AS (
            SELECT 
                c_custkey,
                c_mktsegment,
                COUNT(DISTINCT o_orderkey) as order_count,
                COUNT(*) as lineitem_count,
                SUM(l_extendedprice * (1 - l_discount)) as total_contribution,
                MAX(l_extendedprice * (1 - l_discount)) as max_item_contribution,
                AVG(l_extendedprice * (1 - l_discount)) as avg_item_contribution
            FROM 
                customer 
                JOIN orders ON c_custkey = o_custkey
                JOIN lineitem ON o_orderkey = l_orderkey
            GROUP BY 
                c_custkey, c_mktsegment
        )
        SELECT 
            c_mktsegment as segment,
            COUNT(*) as customer_count,
            AVG(order_count) as avg_orders_per_customer,
            AVG(lineitem_count) as avg_items_per_customer,
            AVG(total_contribution) as avg_contribution_per_customer,
            MAX(total_contribution) as max_contribution_per_customer
        FROM 
            customer_stats
        GROUP BY 
            c_mktsegment
        ORDER BY 
            avg_contribution_per_customer DESC
        """)
        
        customer_base = pd.read_sql(query, self.engine)
        
        # 在Python中计算分位数
        segments = customer_base['segment'].unique()
        enhanced_customer_stats = []
        
        for segment in segments:
            # 获取该细分下所有客户的总贡献
            contribution_query = text(f"""
            SELECT SUM(l_extendedprice * (1 - l_discount)) as total_contribution
            FROM customer 
            JOIN orders ON c_custkey = o_custkey
            JOIN lineitem ON o_orderkey = l_orderkey
            WHERE c_mktsegment = '{segment}'
            GROUP BY c_custkey
            """)
            
            contributions = pd.read_sql(contribution_query, self.engine)
            
            segment_row = customer_base[customer_base['segment'] == segment].iloc[0].copy()
            segment_row['p25_contribution'] = contributions['total_contribution'].quantile(0.25)
            segment_row['p50_contribution'] = contributions['total_contribution'].quantile(0.50)
            segment_row['p75_contribution'] = contributions['total_contribution'].quantile(0.75)
            segment_row['p90_contribution'] = contributions['total_contribution'].quantile(0.90)
            
            enhanced_customer_stats.append(segment_row)
        
        customer_patterns = pd.DataFrame(enhanced_customer_stats)
        logger.info(f"分析了客户行为模式，涵盖 {len(customer_patterns)} 个细分市场")
        return customer_patterns
    
    def analyze_specific_query_pattern(self, market_segment, date):
        """分析特定查询模式的数据特征 - 修复MySQL语法"""
        logger.info(f"分析查询模式: {market_segment} {date}")
        
        # 获取基础统计
        query = text(f"""
        WITH filtered_data AS (
            SELECT 
                c.c_custkey,
                l.l_orderkey,
                l.l_linenumber,
                l.l_extendedprice,
                l.l_discount,
                (l.l_extendedprice * (1 - l.l_discount)) as contribution
            FROM 
                customer c
                JOIN orders o ON c.c_custkey = o.o_custkey
                JOIN lineitem l ON o.o_orderkey = l.l_orderkey
            WHERE 
                c.c_mktsegment = '{market_segment}'
                AND o.o_orderdate < '{date}'
                AND l.l_shipdate > '{date}'
        ),
        customer_aggregates AS (
            SELECT 
                c_custkey,
                COUNT(*) as item_count,
                SUM(contribution) as total_contribution,
                MAX(contribution) as max_item_contribution,
                AVG(contribution) as avg_item_contribution
            FROM 
                filtered_data
            GROUP BY 
                c_custkey
        )
        SELECT 
            COUNT(DISTINCT c_custkey) as customer_count,
            COUNT(*) as total_items,
            AVG(item_count) as avg_items_per_customer,
            MAX(item_count) as max_items_per_customer,
            AVG(total_contribution) as avg_contribution_per_customer,
            MAX(total_contribution) as max_contribution_per_customer,
            STDDEV(total_contribution) as std_contribution,
            AVG(avg_item_contribution) as avg_item_value,
            MAX(max_item_contribution) as max_item_value
        FROM 
            customer_aggregates
        """)
        
        try:
            pattern_stats = pd.read_sql(query, self.engine)
            if pattern_stats.empty:
                return None
            
            # 获取分位数数据
            quantile_query = text(f"""
            SELECT 
                c_custkey,
                COUNT(*) as item_count,
                SUM(l_extendedprice * (1 - l_discount)) as total_contribution
            FROM 
                customer c
                JOIN orders o ON c.c_custkey = o.o_custkey
                JOIN lineitem l ON o.o_orderkey = l.l_orderkey
            WHERE 
                c.c_mktsegment = '{market_segment}'
                AND o.o_orderdate < '{date}'
                AND l.l_shipdate > '{date}'
            GROUP BY 
                c_custkey
            """)
            
            quantile_data = pd.read_sql(quantile_query, self.engine)
            
            # 在Python中计算分位数
            pattern_stats = pattern_stats.iloc[0].copy()
            pattern_stats['segment'] = market_segment
            pattern_stats['date'] = date
            pattern_stats['p10_items'] = quantile_data['item_count'].quantile(0.10)
            pattern_stats['p25_items'] = quantile_data['item_count'].quantile(0.25)
            pattern_stats['p50_items'] = quantile_data['item_count'].quantile(0.50)
            pattern_stats['p75_items'] = quantile_data['item_count'].quantile(0.75)
            pattern_stats['p90_items'] = quantile_data['item_count'].quantile(0.90)
            pattern_stats['p10_contribution'] = quantile_data['total_contribution'].quantile(0.10)
            pattern_stats['p25_contribution'] = quantile_data['total_contribution'].quantile(0.25)
            pattern_stats['p50_contribution'] = quantile_data['total_contribution'].quantile(0.50)
            pattern_stats['p75_contribution'] = quantile_data['total_contribution'].quantile(0.75)
            pattern_stats['p90_contribution'] = quantile_data['total_contribution'].quantile(0.90)
            
            # 计算复杂度指标
            pattern_stats['complexity_score'] = self._calculate_dynamic_complexity(pattern_stats)
            pattern_stats['data_richness'] = self._calculate_data_richness(pattern_stats)
            
            logger.info(f"查询模式分析完成: {market_segment} {date}")
            return pattern_stats
            
        except Exception as e:
            logger.error(f"分析查询模式失败 {market_segment} {date}: {e}")
            return None
    
    def _calculate_dynamic_complexity(self, stats):
        """动态计算数据复杂度"""
        complexity = 0.0
        
        # 客户数量影响 (0-0.3)
        customer_factor = min(stats['customer_count'] / 200.0, 1.5)
        complexity += customer_factor * 0.3
        
        # 数据分布不均匀性 (0-0.4)
        if stats['avg_contribution_per_customer'] > 0:
            cv = stats['std_contribution'] / stats['avg_contribution_per_customer']
            uneven_factor = min(cv, 2.0) / 2.0
            complexity += uneven_factor * 0.4
        
        # 极端值影响 (0-0.3)
        if stats['avg_contribution_per_customer'] > 0:
            outlier_factor = min(stats['max_contribution_per_customer'] / stats['avg_contribution_per_customer'] / 5.0, 1.0)
            complexity += outlier_factor * 0.3
        
        return min(complexity, 1.0)
    
    def _calculate_data_richness(self, stats):
        """计算数据丰富度"""
        richness = 0.0
        
        # 客户密度 (0-0.4)
        customer_density = min(stats['customer_count'] / 100.0, 2.0)
        richness += customer_density * 0.4
        
        # 数据点密度 (0-0.3)
        item_density = min(stats['total_items'] / 500.0, 2.0)
        richness += item_density * 0.3
        
        # 价值分布广度 (0-0.3)
        if stats['avg_item_value'] > 0:
            value_range = min(stats['max_item_value'] / stats['avg_item_value'] / 10.0, 1.0)
            richness += value_range * 0.3
        
        return min(richness, 1.0)