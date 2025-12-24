#!/usr/bin/env python3
"""
部署管理器
提供LTV预测技能的部署、集成和管理功能
"""

import os
import json
import pickle
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import zipfile
from datetime import datetime

class DeploymentManager:
    """部署管理器类"""

    def __init__(self, skill_dir: str = None):
        """
        初始化部署管理器

        Args:
            skill_dir: 技能根目录
        """
        if skill_dir is None:
            # 默认为当前脚本所在目录的上级目录
            self.skill_dir = Path(__file__).parent.parent
        else:
            self.skill_dir = Path(skill_dir)

        self.config_dir = self.skill_dir / 'config'
        self.models_dir = self.skill_dir / 'models'
        self.data_dir = self.skill_dir / 'data'
        self.docs_dir = self.skill_dir / 'docs'

        # 确保目录存在
        for dir_path in [self.config_dir, self.models_dir, self.data_dir, self.docs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_deployment_package(self, output_path: str = None,
                                include_models: bool = True,
                                include_data: bool = False,
                                include_docs: bool = True) -> str:
        """
        创建部署包

        Args:
            output_path: 输出路径
            include_models: 是否包含模型文件
            include_data: 是否包含数据文件
            include_docs: 是否包含文档

        Returns:
            部署包路径
        """
        print("📦 创建部署包...")

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"ltv_predictor_deployment_{timestamp}.zip"

        output_path = Path(output_path)

        # 创建临时目录
        temp_dir = self.skill_dir / 'temp_deployment'
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        try:
            # 复制核心文件
            self._copy_core_files(temp_dir)

            # 复制模型文件（如果需要）
            if include_models:
                self._copy_models(temp_dir)

            # 复制数据文件（如果需要）
            if include_data:
                self._copy_data(temp_dir)

            # 复制文档（如果需要）
            if include_docs:
                self._copy_docs(temp_dir)

            # 创建部署配置
            self._create_deployment_config(temp_dir)

            # 创建启动脚本
            self._create_startup_scripts(temp_dir)

            # 创建ZIP包
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)

            print(f"✅ 部署包已创建: {output_path}")
            print(f"   包大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        finally:
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        return str(output_path)

    def _copy_core_files(self, temp_dir: Path):
        """复制核心文件"""
        print("  复制核心文件...")

        core_files = [
            'scripts/data_processor.py',
            'scripts/regression_models.py',
            'scripts/ltv_predictor.py',
            'scripts/visualizer.py',
            'scripts/report_generator.py',
            'scripts/quick_analysis.py',
            'scripts/model_optimizer.py',
            'scripts/advanced_analytics.py',
            'scripts/deployment_manager.py',
            'SKILL.md',
            'README.md'
        ]

        for file_path in core_files:
            src = self.skill_dir / file_path
            if src.exists():
                dst = temp_dir / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"    ✓ {file_path}")

    def _copy_models(self, temp_dir: Path):
        """复制模型文件"""
        print("  复制模型文件...")

        models_src = self.skill_dir / 'models'
        if models_src.exists():
            models_dst = temp_dir / 'models'
            shutil.copytree(models_src, models_dst, dirs_exist_ok=True)
            print(f"    ✓ 模型文件已复制")

    def _copy_data(self, temp_dir: Path):
        """复制数据文件"""
        print("  复制数据文件...")

        data_src = self.skill_dir / 'data'
        if data_src.exists():
            data_dst = temp_dir / 'data'
            shutil.copytree(data_src, data_dst, dirs_exist_ok=True)
            print(f"    ✓ 数据文件已复制")

    def _copy_docs(self, temp_dir: Path):
        """复制文档文件"""
        print("  复制文档文件...")

        docs_src = self.skill_dir / 'docs'
        if docs_src.exists():
            docs_dst = temp_dir / 'docs'
            shutil.copytree(docs_src, docs_dst, dirs_exist_ok=True)
            print(f"    ✓ 文档文件已复制")

        examples_src = self.skill_dir / 'examples'
        if examples_src.exists():
            examples_dst = temp_dir / 'examples'
            shutil.copytree(examples_src, examples_dst, dirs_exist_ok=True)
            print(f"    ✓ 示例文件已复制")

    def _create_deployment_config(self, temp_dir: Path):
        """创建部署配置"""
        config = {
            'skill_info': {
                'name': 'ltv-predictor',
                'version': '1.0.0',
                'description': '客户生命周期价值预测技能',
                'author': 'Claude Code',
                'created_at': datetime.now().isoformat()
            },
            'dependencies': [
                'pandas>=1.3.0',
                'numpy>=1.21.0',
                'scikit-learn>=1.0.0',
                'matplotlib>=3.5.0',
                'seaborn>=0.11.0',
                'openpyxl>=3.0.0'
            ],
            'default_config': {
                'feature_period_months': 3,
                'prediction_period_months': 12,
                'models_to_train': ['linear_regression', 'random_forest'],
                'enable_visualization': True,
                'enable_reports': True
            },
            'api_endpoints': {
                'analyze': '/api/v1/analyze',
                'predict': '/api/v1/predict',
                'batch_predict': '/api/v1/batch_predict',
                'model_info': '/api/v1/model_info'
            }
        }

        config_path = temp_dir / 'deployment_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"    ✓ 部署配置已创建")

    def _create_startup_scripts(self, temp_dir: Path):
        """创建启动脚本"""
        # 创建Windows批处理文件
        windows_script = '''@echo off
echo 启动LTV预测技能服务...
python scripts/deployment_manager.py start_server
pause
'''
        with open(temp_dir / 'start.bat', 'w', encoding='utf-8') as f:
            f.write(windows_script)

        # 创建Linux/Mac shell脚本
        linux_script = '''#!/bin/bash
echo "启动LTV预测技能服务..."
python scripts/deployment_manager.py start_server
'''
        with open(temp_dir / 'start.sh', 'w', encoding='utf-8') as f:
            f.write(linux_script)

        # 使shell脚本可执行
        os.chmod(temp_dir / 'start.sh', 0o755)

        print(f"    ✓ 启动脚本已创建")

    def validate_deployment(self, deployment_path: str) -> Dict[str, Any]:
        """
        验证部署包

        Args:
            deployment_path: 部署包路径

        Returns:
            验证结果
        """
        print("🔍 验证部署包...")

        deployment_path = Path(deployment_path)
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_count': 0,
            'total_size': 0
        }

        try:
            # 检查文件是否存在
            if not deployment_path.exists():
                validation_result['is_valid'] = False
                validation_result['errors'].append("部署包文件不存在")
                return validation_result

            # 检查文件大小
            size_mb = deployment_path.stat().st_size / 1024 / 1024
            validation_result['total_size'] = size_mb

            if size_mb > 100:
                validation_result['warnings'].append("部署包较大，建议优化大小")

            # 检查ZIP文件完整性
            with zipfile.ZipFile(deployment_path, 'r') as zipf:
                validation_result['file_count'] = len(zipf.namelist())

                # 检查必要文件
                required_files = [
                    'scripts/ltv_predictor.py',
                    'scripts/quick_analysis.py',
                    'deployment_config.json'
                ]

                for required_file in required_files:
                    if required_file not in zipf.namelist():
                        validation_result['is_valid'] = False
                        validation_result['errors'].append(f"缺少必要文件: {required_file}")

            print(f"✅ 部署包验证完成")
            print(f"   文件数量: {validation_result['file_count']}")
            print(f"   包大小: {size_mb:.1f} MB")
            print(f"   验证状态: {'通过' if validation_result['is_valid'] else '失败'}")

            if validation_result['warnings']:
                print(f"   警告: {len(validation_result['warnings'])}个")

        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"验证过程中出现错误: {str(e)}")

        return validation_result

    def install_dependencies(self) -> bool:
        """
        安装依赖包

        Returns:
            是否安装成功
        """
        print("📦 安装依赖包...")

        dependencies = [
            'pandas>=1.3.0',
            'numpy>=1.21.0',
            'scikit-learn>=1.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'openpyxl>=3.0.0'
        ]

        try:
            import subprocess
            for dep in dependencies:
                print(f"  安装 {dep}...")
                result = subprocess.run(['pip', 'install', dep],
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"    ❌ 安装失败: {result.stderr}")
                    return False
                print(f"    ✓ {dep}")

            print("✅ 所有依赖包安装成功")
            return True

        except Exception as e:
            print(f"❌ 依赖包安装失败: {str(e)}")
            return False

    def setup_environment(self) -> bool:
        """
        设置运行环境

        Returns:
            是否设置成功
        """
        print("🔧 设置运行环境...")

        try:
            # 创建必要的目录
            directories = [
                'logs',
                'temp',
                'output',
                'models/trained',
                'data/upload',
                'reports'
            ]

            for directory in directories:
                dir_path = self.skill_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"    ✓ 创建目录: {directory}")

            # 创建环境配置文件
            env_config = {
                'LOG_LEVEL': 'INFO',
                'MAX_UPLOAD_SIZE': '100MB',
                'DEFAULT_MODEL': 'random_forest',
                'CACHE_TIMEOUT': 3600,
                'ENABLE_MONITORING': True
            }

            env_path = self.skill_dir / '.env'
            with open(env_path, 'w', encoding='utf-8') as f:
                for key, value in env_config.items():
                    f.write(f"{key}={value}\n")

            print("    ✓ 环境配置已创建")

            print("✅ 运行环境设置完成")
            return True

        except Exception as e:
            print(f"❌ 环境设置失败: {str(e)}")
            return False

    def create_api_server(self) -> str:
        """
        创建API服务器脚本

        Returns:
            API服务器脚本路径
        """
        print("🌐 创建API服务器...")

        api_script = '''#!/usr/bin/env python3
"""
LTV预测技能API服务器
提供RESTful API接口
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加技能模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'scripts'))

try:
    from flask import Flask, request, jsonify
    from quick_analysis import quick_ltv_analysis, predict_new_customers
    from ltv_predictor import LTVPredictor
except ImportError as e:
    print(f"缺少依赖包: {e}")
    print("请运行: pip install flask")
    sys.exit(1)

app = Flask(__name__)

# 全局变量
predictor = None

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/v1/analyze', methods=['POST'])
def analyze_data():
    """分析数据并训练模型"""
    try:
        data = request.get_json()

        # 验证输入参数
        if 'data_file' not in data:
            return jsonify({'error': '缺少data_file参数'}), 400

        # 执行分析
        results = quick_ltv_analysis(
            file_path=data['data_file'],
            feature_period_months=data.get('feature_period_months', 3),
            prediction_period_months=data.get('prediction_period_months', 12),
            output_dir=data.get('output_dir', './api_results'),
            generate_charts=data.get('generate_charts', True),
            generate_reports=data.get('generate_reports', True)
        )

        return jsonify({
            'status': 'success',
            'results': {
                'best_model': results['summary']['model_summary']['best_model'],
                'r2_score': results['summary']['model_summary']['best_r2_score'],
                'total_customers': results['summary']['data_summary']['total_customers'],
                'avg_ltv': results['summary']['data_summary']['avg_ltv']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/predict', methods=['POST'])
def predict_ltv():
    """预测新客户LTV"""
    try:
        data = request.get_json()

        if 'model_dir' not in data or 'new_orders_file' not in data:
            return jsonify({'error': '缺少必要参数'}), 400

        # 执行预测
        predictions = predict_new_customers(
            data['model_dir'],
            data['new_orders_file'],
            data.get('output_path', 'api_predictions.csv')
        )

        return jsonify({
            'status': 'success',
            'predictions_count': len(predictions),
            'avg_predicted_ltv': predictions['预测LTV'].mean(),
            'max_predicted_ltv': predictions['预测LTV'].max(),
            'min_predicted_ltv': predictions['预测LTV'].min()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/model_info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    return jsonify({
        'available_models': ['linear_regression', 'random_forest'],
        'default_features': ['R值', 'F值', 'M值'],
        'supported_formats': ['csv', 'xlsx'],
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 启动LTV预测API服务器...")
    print("📡 服务地址: http://localhost:5000")
    print("📖 API文档: http://localhost:5000/api/v1/model_info")

    app.run(host='0.0.0.0', port=5000, debug=True)
'''

        api_path = self.skill_dir / 'api_server.py'
        with open(api_path, 'w', encoding='utf-8') as f:
            f.write(api_script)

        print(f"✅ API服务器脚本已创建: {api_path}")
        return str(api_path)

    def start_server(self, host='localhost', port=5000):
        """
        启动API服务器

        Args:
            host: 主机地址
            port: 端口号
        """
        print(f"🚀 启动服务器 {host}:{port}...")

        try:
            # 检查Flask是否安装
            import flask
        except ImportError:
            print("❌ 缺少Flask依赖包")
            print("请运行: pip install flask")
            return

        # 创建并启动API服务器
        api_path = self.create_api_server()
        os.system(f"python {api_path}")

    def generate_deployment_guide(self, output_path: str = None) -> str:
        """
        生成部署指南

        Args:
            output_path: 输出路径

        Returns:
            部署指南路径
        """
        print("📖 生成部署指南...")

        if output_path is None:
            output_path = self.skill_dir / 'DEPLOYMENT_GUIDE.md'

        guide = '''# LTV预测技能部署指南

## 系统要求

- Python 3.7+
- 内存: 最少2GB，推荐4GB+
- 存储: 最少1GB可用空间

## 快速部署

### 1. 环境准备

```bash
# 安装Python依赖
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl flask

# 或使用requirements.txt
pip install -r requirements.txt
```

### 2. 部署包部署

```bash
# 解压部署包
unzip ltv_predictor_deployment_*.zip
cd ltv_predictor

# 安装依赖
python scripts/deployment_manager.py install_dependencies

# 设置环境
python scripts/deployment_manager.py setup_environment
```

### 3. 启动服务

#### 方式1: 命令行工具
```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

#### 方式2: Python脚本
```bash
python scripts/deployment_manager.py start_server
```

#### 方式3: 直接启动API
```bash
python api_server.py
```

## API接口

### 健康检查
```http
GET /api/v1/health
```

### 数据分析
```http
POST /api/v1/analyze
Content-Type: application/json

{
  "data_file": "path/to/your/data.csv",
  "feature_period_months": 3,
  "prediction_period_months": 12,
  "output_dir": "./results"
}
```

### LTV预测
```http
POST /api/v1/predict
Content-Type: application/json

{
  "model_dir": "./models",
  "new_orders_file": "new_customers.csv",
  "output_path": "predictions.csv"
}
```

## 使用示例

### Python客户端
```python
import requests

# 分析数据
response = requests.post('http://localhost:5000/api/v1/analyze', json={
    'data_file': 'data/orders.csv'
})
result = response.json()

# 预测LTV
response = requests.post('http://localhost:5000/api/v1/predict', json={
    'model_dir': './models',
    'new_orders_file': 'new_customers.csv'
})
predictions = response.json()
```

### 命令行工具
```bash
# 基础分析
python scripts/quick_analysis.py analyze data/orders.csv

# 预测新客户
python scripts/quick_analysis.py predict ./models new_customers.csv

# 批量预测
python scripts/quick_analysis.py batch ./models rfm_features.csv
```

## 配置说明

### 环境变量
- `LOG_LEVEL`: 日志级别 (DEBUG/INFO/WARNING/ERROR)
- `MAX_UPLOAD_SIZE`: 最大上传文件大小
- `DEFAULT_MODEL`: 默认模型 (linear_regression/random_forest)
- `ENABLE_MONITORING`: 是否启用监控

### 配置文件
编辑 `deployment_config.json` 文件来自定义配置。

## 故障排除

### 常见问题

1. **依赖包安装失败**
   ```bash
   # 使用国内镜像
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ 包名
   ```

2. **内存不足**
   - 减少数据集大小
   - 调整模型参数
   - 增加系统内存

3. **模型训练慢**
   - 启用并行处理: `n_jobs=-1`
   - 减少交叉验证折数
   - 使用更简单的模型

4. **API服务无法启动**
   - 检查端口是否被占用
   - 确认Flask依赖已安装
   - 查看错误日志

### 日志查看
```bash
# 查看应用日志
tail -f logs/application.log

# 查看错误日志
tail -f logs/error.log
```

## 性能优化

### 数据优化
- 使用CSV格式而不是Excel
- 预处理数据去除异常值
- 合理设置时间窗口

### 模型优化
- 启用超参数调优
- 使用特征选择
- 考虑模型集成

### 系统优化
- 增加内存配置
- 使用SSD存储
- 启用缓存机制

## 监控和维护

### 性能监控
- API响应时间
- 模型预测准确性
- 系统资源使用率

### 定期维护
- 更新依赖包版本
- 重新训练模型
- 清理临时文件

## 技术支持

如遇到问题，请：
1. 查看本文档的故障排除部分
2. 检查日志文件获取详细错误信息
3. 提交Issue或联系技术支持

---

**版本**: 1.0.0
**更新时间**: {datetime.now().strftime('%Y-%m-%d')}
'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(guide)

        print(f"✅ 部署指南已创建: {output_path}")
        return str(output_path)

def main():
    """主函数 - 命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description='LTV预测技能部署管理器')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 创建部署包命令
    package_parser = subparsers.add_parser('package', help='创建部署包')
    package_parser.add_argument('--output', '-o', help='输出文件路径')
    package_parser.add_argument('--no-models', action='store_true', help='不包含模型文件')
    package_parser.add_argument('--include-data', action='store_true', help='包含数据文件')
    package_parser.add_argument('--no-docs', action='store_true', help='不包含文档')

    # 验证部署包命令
    validate_parser = subparsers.add_parser('validate', help='验证部署包')
    validate_parser.add_argument('package_path', help='部署包路径')

    # 安装依赖命令
    subparsers.add_parser('install', help='安装依赖包')

    # 设置环境命令
    subparsers.add_parser('setup', help='设置运行环境')

    # 启动服务器命令
    server_parser = subparsers.add_parser('start_server', help='启动API服务器')
    server_parser.add_argument('--host', default='localhost', help='主机地址')
    server_parser.add_argument('--port', type=int, default=5000, help='端口号')

    # 生成指南命令
    guide_parser = subparsers.add_parser('guide', help='生成部署指南')
    guide_parser.add_argument('--output', '-o', help='输出文件路径')

    args = parser.parse_args()

    if args.command == 'package':
        # 创建部署包
        manager = DeploymentManager()
        manager.create_deployment_package(
            output_path=args.output,
            include_models=not args.no_models,
            include_data=args.include_data,
            include_docs=not args.no_docs
        )

    elif args.command == 'validate':
        # 验证部署包
        manager = DeploymentManager()
        result = manager.validate_deployment(args.package_path)
        print(f"验证结果: {'通过' if result['is_valid'] else '失败'}")
        if result['errors']:
            print("错误:")
            for error in result['errors']:
                print(f"  - {error}")
        if result['warnings']:
            print("警告:")
            for warning in result['warnings']:
                print(f"  - {warning}")

    elif args.command == 'install':
        # 安装依赖
        manager = DeploymentManager()
        success = manager.install_dependencies()
        if success:
            print("依赖安装成功")
        else:
            print("依赖安装失败")

    elif args.command == 'setup':
        # 设置环境
        manager = DeploymentManager()
        success = manager.setup_environment()
        if success:
            print("环境设置成功")
        else:
            print("环境设置失败")

    elif args.command == 'start_server':
        # 启动服务器
        manager = DeploymentManager()
        manager.start_server(args.host, args.port)

    elif args.command == 'guide':
        # 生成指南
        manager = DeploymentManager()
        guide_path = manager.generate_deployment_guide(args.output)
        print(f"部署指南已生成: {guide_path}")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()