# src/tpch_data_loader.py
import pandas as pd
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TPCHDataLoader:
    """支持TPC-H标准参数的数据加载器"""
    
    def __init__(self, db_connection_string):
        self.engine = create_engine(db_connection_string)
        # TPC-H标准参数
        self.market_segments = ['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE']
    
    def get_ground_truth(self, market_segment='BUILDING', date='1995-03-15'):
        """获取指定参数的TPC-H Q3精确结果"""
        query = text(f"""
        SELECT
            l_orderkey,
            SUM(l_extendedprice * (1 - l_discount)) as revenue,
            o_orderdate,
            o_shippriority
        FROM
            customer,
            orders,
            lineitem
        WHERE
            c_mktsegment = '{market_segment}'
            AND c_custkey = o_custkey
            AND l_orderkey = o_orderkey
            AND o_orderdate < '{date}'
            AND l_shipdate > '{date}'
        GROUP BY
            l_orderkey, o_orderdate, o_shippriority
        ORDER BY
            revenue DESC,
            o_orderdate
        LIMIT 10;
        """)
        
        try:
            ground_truth = pd.read_sql(query, self.engine)
            logger.info(f"获取到 {market_segment}-{date} 的黄金标准数据，共{len(ground_truth)}条记录")
            return ground_truth
        except Exception as e:
            logger.error(f"获取黄金标准数据失败: {e}")
            raise
    
    def get_customer_contributions(self, market_segment='BUILDING', date='1995-03-15'):
        """获取指定参数的客户贡献数据"""
        query = text(f"""
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
        ORDER BY c.c_custkey, l.l_orderkey, l.l_linenumber;
        """)
        
        try:
            contributions = pd.read_sql(query, self.engine)
            logger.info(f"获取到 {market_segment}-{date} 的客户贡献数据，"
                       f"共{len(contributions)}条记录，涉及{contributions['c_custkey'].nunique()}个客户")
            return contributions
        except Exception as e:
            logger.error(f"获取客户贡献数据失败: {e}")
            raise
    
    def get_max_value_per_item(self):
        """获取单个订单项的最大可能价值"""
        query = text("""
        SELECT MAX(l_extendedprice * (1 - l_discount)) as max_value
        FROM lineitem;
        """)
        
        result = pd.read_sql(query, self.engine)
        max_val = result['max_value'].iloc[0]
        logger.info(f"单个订单项最大价值: {max_val:.2f}")
        return max_val
    
    def generate_tpch_dates(self, base_date='1995-03-15', num_dates=5):
        """生成TPC-H标准日期范围"""
        base = pd.to_datetime(base_date)
        dates = [base + pd.Timedelta(days=i) for i in range(num_dates)]
        return [d.strftime('%Y-%m-%d') for d in dates]