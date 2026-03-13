---
name: finish-pr-comments
description: Commit+push PR comment fixes and reply to each reviewer comment on the PR
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob
---

# Finish PR Comments Skill

After PR review fixes have been applied and tested, this skill commits the changes, pushes to the existing PR branch, and replies to each reviewer comment.

## Context

Current branch: !`git branch --show-current`
Uncommitted changes: !`git status --short`
Open PRs for this branch: !`gh pr list --head "$(git branch --show-current)" --json number,title,url --jq '.[] | "#\(.number) \(.title) \(.url)"' 2>/dev/null || echo "(none found)"`

## Instructions

### Step 1 — Commit and push

1. Run `git status` to check for uncommitted changes.
2. If there ARE uncommitted changes:
   - Run `git diff` to understand what changed.
   - Stage relevant files (do NOT stage `.claude/settings.local.json` or files containing secrets).
   - Extract the issue number from the branch name (`issue/{N}-...`).
   - Commit with message: `Address PR review feedback (#{N})`
   - The commit message must end with `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
3. If there are NO uncommitted changes, skip to Step 2.
4. Push to the branch: `git push`

### Step 2 — Fetch review comments

1. Determine the PR number from the context above.
2. Fetch all PR review comments (inline):
   `gh api repos/{owner}/{repo}/pulls/{pr_number}/comments --paginate --jq '.[] | {id, user: .user.login, path: .path, line: .line, body: .body}'`
3. Fetch PR issue comments (general discussion):
   `gh api repos/{owner}/{repo}/issues/{pr_number}/comments --paginate --jq '.[] | {id, user: .user.login, body: .body}'`
4. Filter to reviewer comments only (skip the PR author and `dotllm-claude-code-bot[bot]`).

### Step 3 — Reply to each comment

For each reviewer comment, post a reply indicating what was done:

- **If the comment was addressed**: Reply with a brief description of the fix (e.g., "Fixed — added bounds check" or "Good catch, extracted to a helper method").
- **If the comment was intentionally skipped**: Reply explaining why (e.g., "Keeping as-is — the allocation is on the cold path and readability matters more here").
- **If the comment is a general review summary** (not actionable): Reply with a brief acknowledgment if warranted, otherwise skip.

**Reply mechanics**:
- For inline review comments, reply in-thread: `gh api repos/{owner}/{repo}/pulls/{pr_number}/comments -X POST --field body="..." --field in_reply_to={comment_id}`
- For issue comments, reply as a new issue comment: `gh api repos/{owner}/{repo}/issues/{pr_number}/comments -X POST --field body="..."`

**Bot identity**: Always post as the bot. Source `gh-app-token.sh`, export `GH_TOKEN="$CLAUDE_GH_TOKEN"`, run the `gh api` command, then `unset GH_TOKEN` — all in a single Bash call (env does not persist between calls). Do NOT ask for permission — just post. If the token script is missing or fails, fall back to posting as the user's own identity.

### Additional arguments

If `$ARGUMENTS` is provided, use it as additional guidance (e.g., "only reply to Gemini comments", "skip replying to nits", "don't post as bot").
