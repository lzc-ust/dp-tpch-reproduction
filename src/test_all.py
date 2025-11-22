import logging
from data_loader import DataLoader
from naive_laplace import NaiveLaplaceMechanism
from r2t import R2TMechanism
from shifted_inverse import ShiftedInverseMechanism
from evaluator import Evaluator
from config import get_db_connection_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_all_mechanisms():
    """测试所有机制"""
    loader = DataLoader(get_db_connection_string())
    ground_truth = loader.get_ground_truth()
    
    print("=" * 60)
    print("TPC-H Q3 差分隐私机制测试")
    print("=" * 60)
    
    print(f"黄金标准数据: {len(ground_truth)} 条记录")
    print(f"总收入: {ground_truth['revenue'].sum():.2f}")
    print()
    
    # 测试朴素拉普拉斯
    print("1. 测试朴素拉普拉斯机制...")
    try:
        naive_mech = NaiveLaplaceMechanism(loader)
        naive_result = naive_mech.run_mechanism(epsilon=1.0, random_state=42)
        evaluator = Evaluator(ground_truth)
        naive_metrics = evaluator.evaluate_all(naive_result, "NaiveLaplace")
        print(f"   结果: {len(naive_result)} 条记录")
        print(f"   评估: {naive_metrics}")
        print("   ✓ 成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print()
    
    # 测试R2T机制
    print("2. 测试R2T机制...")
    try:
        r2t_mech = R2TMechanism(loader)
        r2t_result = r2t_mech.run_mechanism(epsilon=1.0, random_state=42)
        evaluator = Evaluator(ground_truth)
        r2t_metrics = evaluator.evaluate_all(r2t_result, "R2T")
        print(f"   结果: {len(r2t_result)} 条记录")
        print(f"   评估: {r2t_metrics}")
        print("   ✓ 成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print()
    
    # 测试Shifted Inverse机制
    print("3. 测试Shifted Inverse机制...")
    try:
        si_mech = ShiftedInverseMechanism(loader)
        si_result = si_mech.run_mechanism(epsilon=1.0, random_state=42)
        evaluator = Evaluator(ground_truth)
        si_metrics = evaluator.evaluate_all(si_result, "ShiftedInverse")
        print(f"   结果: {len(si_result)} 条记录")
        print(f"   评估: {si_metrics}")
        print("   ✓ 成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print()
    print("测试完成!")

if __name__ == "__main__":
    test_all_mechanisms()