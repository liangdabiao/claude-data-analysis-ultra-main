# -*- coding: utf-8 -*-
"""
完整数据分析主脚本
自动运行 Stage 1-6 的完整工作流程
"""
import os
import sys
import subprocess
import time
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_stage(script_name, stage_num, stage_name):
    """运行单个阶段"""
    log(f"{'='*60}")
    log(f"运行 Stage {stage_num}: {stage_name}")
    log(f"{'='*60}")

    script_path = os.path.join(os.path.dirname(__file__), script_name)

    try:
        start_time = time.time()

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        elapsed_time = time.time() - start_time

        # 输出
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}", file=sys.stderr)

        if result.returncode == 0:
            log(f"✅ Stage {stage_num} 完成! 耗时: {elapsed_time:.1f}秒")
            return True
        else:
            log(f"❌ Stage {stage_num} 失败! 退出码: {result.returncode}")
            return False

    except Exception as e:
        log(f"❌ Stage {stage_num} 异常: {str(e)}")
        return False

def main():
    log("="*60)
    log("🚀 Olist 电商数据 - 完整分析流程")
    log("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log(f"工作目录: {base_dir}")

    # 检查数据文件
    data_dir = os.path.join(base_dir, "..", "data_storage")
    required_files = [
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_customers_dataset.csv"
    ]

    missing_files = []
    for f in required_files:
        if not os.path.exists(os.path.join(data_dir, f)):
            missing_files.append(f)

    if missing_files:
        log(f"⚠️ 警告: 缺失数据文件: {missing_files}")
        log(f"请确保数据文件在 {data_dir}/ 目录中")
    else:
        log("✅ 数据文件检查通过")

    # 创建workflow_log目录
    log_dir = os.path.join(base_dir, "workflow_log")
    os.makedirs(log_dir, exist_ok=True)

    # 记录总开始时间
    total_start = time.time()

    # 运行所有阶段
    stages = [
        ("stage1_data_quality.py", 1, "数据质量检查"),
        ("stage2_eda.py", 2, "探索性数据分析"),
        ("stage3_hypothesis.py", 3, "研究假设生成"),
        ("stage4_visualization.py", 4, "数据可视化"),
        ("stage5_codegen.py", 5, "代码生成"),
        ("stage6_report.py", 6, "综合报告生成"),
    ]

    results = []
    for script, num, name in stages:
        success = run_stage(script, num, name)
        results.append((num, name, success))

    # 总耗时
    total_time = time.time() - total_start

    # 生成最终总结
    log("\n" + "="*60)
    log("📊 完整分析执行总结")
    log("="*60)

    for num, name, success in results:
        status = "✅" if success else "❌"
        log(f"  {status} Stage {num}: {name}")

    all_success = all(r[2] for r in results)

    log("")
    log(f"总耗时: {total_time:.1f}秒")

    if all_success:
        log("🎉 恭喜！完整分析成功完成！")
    else:
        log("⚠️ 部分阶段执行失败，请检查日志")

    # 输出文件列表
    log("\n📁 生成的文件列表:")
    log("  - data_quality_report/ - 数据质量报告")
    log("  - exploratory_analysis/ - 探索性分析")
    log("  - hypothesis_reports/ - 研究假设")
    log("  - visualizations/ - 可视化")
    log("  - generated_code/ - 生成的代码")
    log("  - final_report/ - 最终报告")
    log("  - workflow_log/ - 执行日志")

    # 关键输出文件提示
    log("\n🔍 快速查看:")
    log("  - 最终报告: final_report/comprehensive_analysis_report.md")
    log("  - 执行摘要: final_report/executive_summary.md")
    log("  - 可视化仪表板: visualizations/interactive_dashboard.html")
    log("  - 分析代码: generated_code/complete_analysis_pipeline.py")

    log("\n✅ 分析流程结束!")

    # 保存执行摘要
    summary_path = os.path.join(log_dir, "execution_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# 完整分析执行摘要\n\n")
        f.write(f"**执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**总耗时**: {total_time:.1f}秒\n\n")
        f.write(f"## 阶段状态\n\n")
        for num, name, success in results:
            status = "✅ 成功" if success else "❌ 失败"
            f.write(f"- Stage {num}: {name} - {status}\n")
        f.write(f"\n## 输出\n\n")
        f.write(f"请查看各子目录的详细结果\n")

    log(f"\n执行摘要已保存: {summary_path}")

if __name__ == "__main__":
    main()
