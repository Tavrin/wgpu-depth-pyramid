#!/usr/bin/env bash
# Scan tracked and untracked files, without depending on repository state.
set -euo pipefail
cd "$(dirname "$0")/.."
private='mo''ss'
planning='rf''c'
pattern="/home/[a-z]+/|/mnt/[a-z]|${private}-(scratch|cargo|heavy|coord|lane|capture)|automation_game|etienne|Legion-Pro|\b${private}\b|${planning} ?[0-9]|${private}-[a-z0-9]{5}|freelancer|harry.?potter"
set +e
rg --hidden --no-ignore -n -i "$pattern" . --glob '!.git/**' --glob '!target/**' --glob '!tools/check_public_text.sh'
status=$?
set -e
case "$status" in
  1) echo 'PASS: public text contains no private machine paths or project names' ;;
  0) echo 'FAIL: private text found' >&2; exit 1 ;;
  *) exit "$status" ;;
esac
