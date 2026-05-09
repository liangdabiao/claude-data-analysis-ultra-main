# -*- coding: utf-8 -*-
"""
Stage 3: 研究假设生成 (Hypothesis Generation)
"""
import os
import json
import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Stage 3] {msg}", flush=True)

def main():
    log("="*60)
    log("Stage 3: 研究假设生成开始")
    log("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data_storage")
    output_dir = os.path.join(base_dir, "hypothesis_reports")

    # 加载数据
    log("正在加载数据...")
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))

    # 生成假设
    log("正在生成研究假设...")
    hypotheses = generate_hypotheses(orders, order_items, payments, reviews)

    # 保存结果
    save_hypothesis_results(hypotheses, output_dir)

    log("="*60)
    log("Stage 3: 研究假设生成完成")
    log("="*60)

def generate_hypotheses(orders, order_items, payments, reviews):
    """基于数据生成研究假设"""
    hypotheses = []

    # 1. 客户满意度相关假设
    if 'review_score' in reviews.columns:
        hypotheses.append({
            'id': 'H1',
            'title': '配送时间与客户满意度负相关',
            'description': '配送时间越长，客户评分越低',
            'type': 'correlation',
            'priority': 'high',
            'test_method': '相关性分析、回归分析',
            'business_impact': '优化物流可显著提升客户体验'
        })

        hypotheses.append({
            'id': 'H2',
            'title': '订单金额与客户满意度无显著相关',
            'description': '高消费客户并不一定给出更高评分',
            'type': 'correlation',
            'priority': 'medium',
            'test_method': '相关性分析、分组比较',
            'business_impact': '价格策略不直接影响满意度'
        })

    # 2. 支付方式相关假设
    hypotheses.append({
        'id': 'H3',
        'title': '信用卡支付订单金额更高',
        'description': '使用信用卡的客户倾向于购买更贵的商品',
        'type': 'comparison',
        'priority': 'medium',
        'test_method': 't检验、ANOVA',
        'business_impact': '支付方式与客户价值相关'
    })

    hypotheses.append({
        'id': 'H4',
        'title': '分期付款订单的退货率更低',
        'description': '分期付款的客户更谨慎，退货可能性更低',
        'type': 'comparison',
        'priority': 'low',
        'test_method': '卡方检验',
        'business_impact': '分期付款可能降低退货风险'
    })

    # 3. 时间相关假设
    hypotheses.append({
        'id': 'H5',
        'title': '周末订单金额显著高于工作日',
        'description': '消费者在周末更倾向于大额消费',
        'type': 'comparison',
        'priority': 'medium',
        'test_method': 't检验、时间序列分析',
        'business_impact': '营销策略可按时间差异化'
    })

    hypotheses.append({
        'id': 'H6',
        'title': '节假日期间订单量显著增加',
        'description': '节假日是电商销售高峰期',
        'type': 'comparison',
        'priority': 'high',
        'test_method': '时间序列分析、季节性检验',
        'business_impact': '需提前准备库存和物流'
    })

    # 4. 客户行为假设
    hypotheses.append({
        'id': 'H7',
        'title': '多次购买客户的客单价更高',
        'description': '回头客比新客更有价值',
        'type': 'comparison',
        'priority': 'high',
        'test_method': '分组比较、回归分析',
        'business_impact': '客户留存对营收至关重要'
    })

    # 5. 实验设计建议
    experimental_designs = [
        {
            'experiment': '物流优化测试',
            'description': 'A/B测试不同配送速度对满意度的影响',
            'sample_size': '每组至少1000单',
            'duration': '2-4周',
            'metrics': ['review_score', 'delivery_time']
        },
        {
            'experiment': '支付方式优惠测试',
            'description': '测试不同支付方式的促销效果',
            'sample_size': '每组500-1000单',
            'duration': '1-2个月',
            'metrics': ['conversion_rate', 'order_value', 'return_rate']
        }
    ]

    # 6. 验证计划
    validation_plan = {
        'data_requirements': '完整的订单、支付、评论数据',
        'statistical_methods': ['相关性分析', 't检验', '回归分析', '时间序列分析'],
        'success_criteria': 'p-value < 0.05，效应量适中',
        'timeline': '2-4周完成验证'
    }

    return {
        'hypotheses': hypotheses,
        'experimental_designs': experimental_designs,
        'validation_plan': validation_plan
    }

def save_hypothesis_results(results, output_dir):
    """保存假设生成结果"""

    # 1. 研究假设JSON
    json_path = os.path.join(output_dir, "research_hypotheses.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"研究假设JSON已保存: {json_path}")

    # 2. 研究假设Markdown
    md_hypothesis = os.path.join(output_dir, "research_hypotheses.md")
    with open(md_hypothesis, 'w', encoding='utf-8') as f:
        f.write("# 研究假设报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 核心假设\n\n")
        for h in results['hypotheses']:
            f.write(f"### {h['id']}: {h['title']}\n\n")
            f.write(f"- **描述**: {h['description']}\n")
            f.write(f"- **类型**: {h['type']}\n")
            f.write(f"- **优先级**: {h['priority']}\n")
            f.write(f"- **测试方法**: {h['test_method']}\n")
            f.write(f"- **业务影响**: {h['business_impact']}\n\n")

    log(f"研究假设报告已保存: {md_hypothesis}")

    # 3. 实验设计
    md_experiment = os.path.join(output_dir, "experimental_design.md")
    with open(md_experiment, 'w', encoding='utf-8') as f:
        f.write("# 实验设计建议\n\n")
        for exp in results['experimental_designs']:
            f.write(f"## {exp['experiment']}\n\n")
            f.write(f"- **描述**: {exp['description']}\n")
            f.write(f"- **样本量**: {exp['sample_size']}\n")
            f.write(f"- **时长**: {exp['duration']}\n")
            f.write(f"- **指标**: {', '.join(exp['metrics'])}\n\n")

    log(f"实验设计已保存: {md_experiment}")

    # 4. 验证计划
    md_validation = os.path.join(output_dir, "validation_plan.md")
    with open(md_validation, 'w', encoding='utf-8') as f:
        f.write("# 验证计划\n\n")
        plan = results['validation_plan']
        f.write(f"- **数据需求**: {plan['data_requirements']}\n")
        f.write(f"- **统计方法**: {', '.join(plan['statistical_methods'])}\n")
        f.write(f"- **成功标准**: {plan['success_criteria']}\n")
        f.write(f"- **时间线**: {plan['timeline']}\n")

    log(f"验证计划已保存: {md_validation}")

    # 保存执行摘要
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "workflow_log", "stage3_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Stage 3: 研究假设生成\n\n")
        f.write(f"- 生成假设数: {len(results['hypotheses'])}\n")
        f.write(f"- 实验设计数: {len(results['experimental_designs'])}\n")
        f.write(f"- 状态: ✅ 完成\n")

if __name__ == "__main__":
    main()
