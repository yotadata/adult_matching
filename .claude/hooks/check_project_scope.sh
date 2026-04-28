#!/usr/bin/env bash
# PreToolUse hook: プロジェクト外へのファイルアクセスをブロックする
python3 "$(dirname "$0")/check_project_scope.py"
