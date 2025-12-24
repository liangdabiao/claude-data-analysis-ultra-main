#!/usr/bin/env python3
"""
Context loader hook for data analysis
Provides additional context based on the current project state
"""

import json
import sys
import os
import glob
from datetime import datetime

def get_project_context():
    """Get current project context"""
    context = []

    # Check for data files
    data_storage = os.path.join(os.getcwd(), 'data_storage')
    if os.path.exists(data_storage):
        data_files = glob.glob(os.path.join(data_storage, '*'))
        if data_files:
            context.append(f"📊 Available datasets: {len(data_files)} files in data_storage/")
            for file in data_files[:5]:  # Show first 5 files
                file_size = os.path.getsize(file) / 1024  # KB
                context.append(f"   - {os.path.basename(file)} ({file_size:.1f} KB)")

    # Check for recent analysis results
    viz_dir = os.path.join(os.getcwd(), 'visualizations')
    if os.path.exists(viz_dir):
        viz_files = glob.glob(os.path.join(viz_dir, '*'))
        if viz_files:
            context.append(f"📈 Generated visualizations: {len(viz_files)} files")

    # Check for recent activity
    log_file = os.path.join(os.path.dirname(__file__), '..', 'validation_log.json')
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            if lines:
                context.append(f"🔍 Recent analysis activity: {len(lines)} validation events")
        except:
            pass

    return context

def main():
    """Main hook function"""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        prompt = input_data.get("prompt", "")

        # Add project context if relevant
        context = get_project_context()

        if context:
            # Add context to the prompt
            context_text = "\n".join(context)
            additional_context = f"\n\n📋 Current Project Context:\n{context_text}"

            # Use JSON output for UserPromptSubmit hook
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": additional_context
                }
            }

            print(json.dumps(output))

        sys.exit(0)

    except Exception as e:
        print(f"Context loader hook error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()