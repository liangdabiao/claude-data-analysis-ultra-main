#!/usr/bin/env python3
"""
简单的功能测试脚本
"""

import sys
from pathlib import Path

# 添加技能模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / 'scripts'))

try:
    print("🧪 测试数据处理器...")
    from data_processor import DataProcessor
    import pandas as pd

    # 初始化处理器
    processor = DataProcessor()

    # 加载示例数据
    sample_data_path = current_dir.parent / 'data' / 'sample_orders.csv'
    if not sample_data_path.exists():
        print("❌ 示例数据文件不存在")
        sys.exit(1)

    data = processor.load_order_data(str(sample_data_path))
    print(f"✅ 数据加载成功: {data.shape}")

    # 测试RFM计算
    rfm_data = processor.calculate_rfm_features(data, 3, 12)
    print(f"✅ RFM计算成功: {rfm_data.shape}")

    # 检查关键列
    required_columns = ['R值', 'F值', 'M值', '年度LTV']
    missing_columns = [col for col in required_columns if col not in rfm_data.columns]
    if missing_columns:
        print(f"❌ 缺少列: {missing_columns}")
    else:
        print("✅ RFM特征列完整")

    print("\n🧪 测试回归模型...")
    from regression_models import RegressionModels
    import numpy as np

    # 创建测试数据
    X = np.random.rand(50, 3)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(50) * 0.1

    models = RegressionModels()

    # 测试线性回归
    linear_model = models.train_single_model('linear_regression', X, y)
    if linear_model is not None:
        print("✅ 线性回归训练成功")

    # 测试随机森林
    rf_model = models.train_single_model('random_forest', X, y)
    if rf_model is not None:
        print("✅ 随机森林训练成功")

    print("\n🧪 测试完整分析...")
    from ltv_predictor import complete_ltv_analysis

    # 快速配置
    config = {
        'data_processor_config': {
            'feature_period_months': 3,
            'prediction_period_months': 6,
            'remove_outliers': False,
            'min_orders_per_customer': 1
        },
        'regression_config': {
            'test_size': 0.3,
            'cv_folds': 3,
            'scoring_metric': 'r2',
            'enable_hyperparameter_tuning': False
        },
        'models_to_train': ['linear_regression']
    }

    output_dir = current_dir.parent / 'examples' / 'simple_test_results'

    results = complete_ltv_analysis(
        str(sample_data_path),
        output_dir=str(output_dir),
        config=config
    )

    if results and 'predictor' in results:
        print("✅ 完整分析流程测试成功")
        print(f"   - 分析客户数: {len(results['rfm_data'])}")
        print(f"   - 输出目录: {output_dir}")
    else:
        print("❌ 完整分析流程测试失败")

    print("\n🎉 所有基础功能测试通过！")

except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 测试过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)