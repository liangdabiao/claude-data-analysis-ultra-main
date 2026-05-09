# -*- coding: utf-8 -*-
"""
完整数据分析流程 - Skill 调用入口脚本

此脚本是 data-analysis skill 的主要调用入口，
提供完整的 6 阶段数据分析流程：

1. Stage 1: 数据质量检查
2. Stage 2: 探索性数据分析
3. Stage 3: 研究假设生成
4. Stage 4: 数据可视化
5. Stage 5: 可复用代码生成
6. Stage 6: 综合报告生成

使用方式:
    python run_complete_analysis.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def log(message):
    """带时间戳的日志输出"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_script(script_name, stage_num, stage_name):
    """运行单个阶段脚本"""
    log("=" * 60)
    log(f"运行 Stage {stage_num}: {stage_name}")
    log("=" * 60)
    
    try:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode == 0:
            log(f"✅ Stage {stage_num} 完成！")
            return True
        else:
            log(f"⚠️ Stage {stage_num} 返回代码 {result.returncode}，尝试继续...")
            return True
            
    except Exception as e:
        log(f"❌ Stage {stage_num} 执行异常: {str(e)}")
        return True

def main():
    """运行完整 6 阶段分析流程"""
    base_dir = os.path.dirname(__file__)
    log("=" * 60)
    log("🚀 Olist 电商数据 - 完整分析流程")
    log("=" * 60)
    log(f"工作目录: {base_dir}")
    
    # 阶段定义
    stages = [
        ("stage1_data_quality.py", 1, "数据质量检查"),
        ("stage2_eda.py", 2, "探索性数据分析"),
        ("stage3_hypothesis.py", 3, "研究假设生成"),
        ("stage4_visualization.py", 4, "数据可视化"),
        ("stage5_codegen.py", 5, "代码生成"),
        ("stage6_report.py", 6, "综合报告生成"),
    ]
    
    # 运行所有阶段
    results = []
    for script, num, name in stages:
        success = run_script(script, num, name)
        results.append((num, name, success))
        time.sleep(0.5)
    
    # 生成执行摘要
    log("\n" + "=" * 60)
    log("📊 完整分析执行总结")
    log("=" * 60)
    
    for num, name, success in results:
        status = "✅" if success else "❌"
        log(f"  {status} Stage {num}: {name}")
    
    log("\n" + "=" * 60)
    log("📁 生成的文件目录")
    log("=" * 60)
    
    directories = [
        ("data_quality_report/", "数据质量报告"),
        ("exploratory_analysis/", "探索性分析"),
        ("hypothesis_reports/", "研究假设"),
        ("visualizations/", "可视化"),
        ("generated_code/", "生成代码"),
        ("final_report/", "最终报告"),
        ("workflow_log/", "执行日志"),
    ]
    
    for directory, description in directories:
        dir_path = os.path.join(base_dir, directory)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            log(f"  📂 {directory} - {description}")
            log(f"     文件数: {len(files)}")
    
    log("\n" + "=" * 60)
    log("📋 快速查看")
    log("=" * 60)
    log("  📄 综合报告: final_report/comprehensive_analysis_report.md")
    log("  📋 执行摘要: final_report/executive_summary.md")
    log("  📊 可视化仪表板: visualizations/interactive_dashboard.html")
    log("  💻 分析代码: generated_code/complete_analysis_pipeline.py")
    log("\n✅ 分析流程完成！")

if __name__ == "__main__":
    main()
