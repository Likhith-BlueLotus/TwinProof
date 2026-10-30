#!/bin/sh
# Deny `git commit` if the test suite or the G7 AST guard fails.
# Muse PreToolUse: payload on stdin, exit 2 = deny; the reason must go to stderr with nothing on stdout,
# or Muse fails open. `--git` skips the payload check (used by .git/hooks/pre-commit).
if [ "$1" != "--git" ]; then
  payload=$(cat)
  case "$payload" in
    *'git commit'*) ;;
    *) exit 0 ;;
  esac
fi
log="${TMPDIR:-/tmp}/twinproof_gate.log"
if ! uv run pytest -q >"$log" 2>&1; then
  echo "blocked: test suite failed; see $log" >&2; exit 2
fi
staged=$(git diff --cached --name-only --diff-filter=ACM -- 'physics/*.py' 'verifier/*.py' 'agents/*.py')
if [ -n "$staged" ] && ! uv run python -m verifier.g7_scan $staged >>"$log" 2>&1; then
  echo "blocked: G7 AST guard rejected staged changes; see $log" >&2; exit 2
fi
exit 0
