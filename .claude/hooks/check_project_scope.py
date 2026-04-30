#!/usr/bin/env python3
"""PreToolUse hook: プロジェクト外へのファイルアクセスをブロックする"""
import json, sys, os

PROJECT_ROOT = os.path.realpath('/Users/akiyukikamiura/Desktop/adult_matching')

d = json.load(sys.stdin)
i = d.get('tool_input', {})

path = i.get('file_path') or i.get('path') or ''

if not path:
    sys.exit(0)

abs_path = os.path.realpath(os.path.abspath(path))

if abs_path != PROJECT_ROOT and not abs_path.startswith(PROJECT_ROOT + os.sep):
    result = {
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': f'プロジェクト外へのアクセスは禁止されています: {path}'
        }
    }
    print(json.dumps(result, ensure_ascii=False))
