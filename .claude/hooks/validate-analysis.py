#!/usr/bin/env python3
"""
Data analysis validation hook
Validates analysis outputs and ensures data quality
"""

import json
import sys
import os
import re
from datetime import datetime

def validate_file_content(file_path, content):
    """Validate the content of analysis files"""
    issues = []

    # Check for common data analysis issues
    if file_path.endswith('.csv'):
        # Validate CSV structure
        lines = content.strip().split('\n')
        if len(lines) < 2:
            issues.append("CSV file appears to be empty or malformed")

        # Check for common CSV issues
        if ',' not in lines[0]:
            issues.append("CSV file may not be properly formatted")

    elif file_path.endswith('.py'):
        # Validate Python code
        dangerous_patterns = [
            r'os\.system\(',
            r'eval\(',
            r'exec\(',
            r'subprocess\.'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, content):
                issues.append(f"Potentially dangerous code pattern found: {pattern}")

    elif file_path.endswith('.json'):
        # Validate JSON structure
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON format: {e}")

    return issues

def main():
    """Main hook function"""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)

        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Only process Write and Edit tools
        if tool_name not in ["Write", "Edit"]:
            sys.exit(0)

        file_path = tool_input.get("file_path", "")
        content = tool_input.get("content", "")

        # Skip validation for certain file types
        skip_extensions = ['.md', '.txt', '.log']
        if any(file_path.endswith(ext) for ext in skip_extensions):
            sys.exit(0)

        # Validate the content
        issues = validate_file_content(file_path, content)

        if issues:
            # Log validation issues
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "issues": issues,
                "tool_name": tool_name
            }

            # Write to validation log
            log_file = os.path.join(os.path.dirname(__file__), '..', 'validation_log.json')
            try:
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except:
                pass  # Don't fail if we can't write to log

            # Output issues to stderr
            for issue in issues:
                print(f"⚠️  Validation Issue: {issue}", file=sys.stderr)

            # Don't block the operation, just warn
            sys.exit(0)

        sys.exit(0)

    except Exception as e:
        print(f"Validation hook error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()