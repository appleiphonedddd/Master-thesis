#!/bin/bash
# push.sh - simple script to add, commit, and push to GitHub

# Stage all changes
git add .

# Commit with message (default: current date if no argument provided)
if [ -z "$1" ]; then
  msg=$(date +"%Y-%m-%d")
else
  msg="$1"
fi
git commit -m "$msg"

# Push to the current branch (main)
git push origin main
echo "Pushed to GitHub with message: $msg"
