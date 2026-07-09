# Launch 24 Cloud Agents (Module 12)

Local agents stop when your laptop sleeps. **Cloud agents run on Cursor VMs.**

## Why in-chat cloud launch failed

This multi-root workspace blocks `environment: "cloud"` in the Task tool. Use the **API** or **Agents Window** in a single-repo folder instead.

## Option A — API (recommended)

```bash
cd "/Users/joshuaterranova/Desktop/Coding Projects/Machine Learning v1/12-field-compounding-validation"
export CURSOR_API_KEY='key_...'   # https://cursor.com/dashboard/api
python3 scripts/launch_cloud_agents.py --from 3   # skip 01-02 already pushed
```

Monitor at https://cursor.com/agents

**Prerequisites:** GitHub integration with read-write on `thatrandomasiandev/field-compounding-validation`.

## Option B — IDE

1. File → Open Folder → `12-field-compounding-validation` only
2. Cmd+Shift+P → Open Agents Window → Cloud
3. One prompt per agent from `MODULE12_AGENT_PLAN.md`

## Done locally (can skip in cloud)

- `agent/01-foundation` — [PR #1](https://github.com/thatrandomasiandev/field-compounding-validation/pull/1)
- `agent/02-utils` — pushed
