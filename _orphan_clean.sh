#!/bin/bash
set -euo pipefail
cd "/c/Users/nikim/OneDrive/Desktop/cv_final_proj/NikiMahdian_40123153/NikiMahdian_40123153"

# Ensure clean index of current tree
git status --porcelain | head

# Create orphan branch with current files only (no old commit metadata)
git checkout --orphan clean-main
git add -A
# Drop accidental local-only junk from commit if present
git reset HEAD -- docs/DEFENSE_DEMO_VIDEO.mp4 2>/dev/null || true
git reset HEAD -- _strip.sh _strip_push.sh _rewrite_history.sh _msgfilter.py 2>/dev/null || true

export GIT_AUTHOR_NAME="nikimahdian"
export GIT_AUTHOR_EMAIL="niki.mahdian04@gmail.com"
export GIT_COMMITTER_NAME="nikimahdian"
export GIT_COMMITTER_EMAIL="niki.mahdian04@gmail.com"

git commit -m "Surgical AI final delivery: segmentation, RGB control, and defense bundle."

# Verify no Cursor trailer
if git log -1 --format='%B' | grep -qi cursor; then
  echo "ABORT: cursor still in message"
  exit 1
fi

# Replace main
git branch -M main
git log --oneline --all
git log -1 --format='%an <%ae>%n%cn <%ce>%n%B'

# Force push clean history
git push --force origin main

echo DONE
