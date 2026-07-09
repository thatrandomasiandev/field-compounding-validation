# Launching 24 Cloud Agents

Cloud agents require **exactly one git remote on the workspace root**. This multi-root Cursor workspace (Curriculum Vitae + 20+ folders) blocks `environment: "cloud"`.

## Fix (pick one)

1. **Open Module 12 as sole workspace:** File → Open Folder → `12-field-compounding-validation`
2. Then in chat: *"Launch all 24 agents from MODULE12_AGENT_PLAN.md as cloud agents"*

Or:

3. Open `Machine Learning v1` as sole workspace **if** you add a git remote at that root (not recommended; modules are separate repos).

## Repo

- GitHub: https://github.com/thatrandomasiandev/field-compounding-validation
- Plan: `MODULE12_AGENT_PLAN.md` (24 workstreams, branches `agent/01-*` … `agent/24-*`)

## Current fallback

24 **local background agents** were launched from the multi-root workspace (same prompts, same branches).

## After agents finish

```bash
git fetch --all
# merge in order from MODULE12_AGENT_PLAN.md
git checkout main && git merge agent/01-foundation
# … through agent/24-paper-observatory
./scripts/run_all_and_validate.sh
```
