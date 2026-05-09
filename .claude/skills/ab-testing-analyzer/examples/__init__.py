"""
AB测试分析技能核心模块

提供专业的AB测试分析功能，包括：
- AB测试设计和分析
- 统计显著性检验
- 用户分群和交互效应分析
- 可视化和报告生成
"""

__version__ = "1.0.0"
__author__ = "Claude Code Skills"

from .ab_test_analyzer import ABTestAnalyzer
from .statistical_tests import StatisticalTests
from .segment_analyzer import SegmentAnalyzer
from .visualizer import ABTestVisualizer

__all__ = [
    'ABTestAnalyzer',
    'StatisticalTests',
    'SegmentAnalyzer',
    'ABTestVisualizer'
]