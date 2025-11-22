import pandas as pd
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, db_connection_string):
        self.engine = create_engine(db_connection_string)
    
    def get_ground_truth(self):
        """获取TPC-H Q3的精确结果作为黄金标准"""
        query = text("""
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
            c_mktsegment = 'BUILDING'
            AND c_custkey = o_custkey
            AND l_orderkey = o_orderkey
            AND o_orderdate < '1995-03-15'
            AND l_shipdate > '1995-03-15'
        GROUP BY
            l_orderkey, o_orderdate, o_shippriority
        ORDER BY
            revenue DESC,
            o_orderdate
        LIMIT 10;
        """)
        
        try:
            ground_truth = pd.read_sql(query, self.engine)
            logger.info(f"获取到黄金标准数据，共{len(ground_truth)}条记录")
            logger.info(f"收入范围: {ground_truth['revenue'].min():.2f} - {ground_truth['revenue'].max():.2f}")
            return ground_truth
        except Exception as e:
            logger.error(f"获取黄金标准数据失败: {e}")
            raise
    
    def get_customer_contributions(self):
        """获取每个客户的所有订单项贡献，用于敏感度计算和高级算法"""
        query = text("""
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
            c.c_mktsegment = 'BUILDING'
            AND o.o_orderdate < '1995-03-15'
            AND l.l_shipdate > '1995-03-15'
        ORDER BY c.c_custkey, l.l_orderkey, l.l_linenumber;
        """)
        
        try:
            contributions = pd.read_sql(query, self.engine)
            logger.info(f"获取到客户贡献数据，共{len(contributions)}条记录，涉及{contributions['c_custkey'].nunique()}个客户")
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

# 使用示例
if __name__ == "__main__":
    from config import get_db_connection_string
    loader = DataLoader(get_db_connection_string())
    ground_truth = loader.get_ground_truth()
    print("黄金标准数据:")
    print(ground_truth)
    
    # 测试客户贡献数据
    contributions = loader.get_customer_contributions()
    print(f"\n客户贡献数据概况:")
    print(f"总记录数: {len(contributions)}")
    print(f"客户数量: {contributions['c_custkey'].nunique()}")
    print(f"订单数量: {contributions['l_orderkey'].nunique()}")
    
    # 计算每个客户的总贡献
    customer_totals = contributions.groupby('c_custkey')['contribution'].sum()
    print(f"\n客户贡献统计:")
    print(f"最大客户贡献: {customer_totals.max():.2f}")
    print(f"平均客户贡献: {customer_totals.mean():.2f}")
    print(f"客户贡献中位数: {customer_totals.median():.2f}")