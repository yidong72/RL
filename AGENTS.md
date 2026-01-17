# AGENTS.md - AI Agent Guidelines for Cluster Development

## Use 'bd' for task tracking"
beads is already installed, you are working across multiple sessions, check bd to continue your work.

## Slurm Cluster Usage Rules

### Resource Limits

- **Maximum concurrent jobs**: 2 jobs with `partition=interactive`
- **Maximum time per job**: 4 hours (`--time=4:00:00`)
- **Always cancel unwanted jobs** before launching new ones to stay within the 2-job limit

### Before Launching Jobs

1. Check current job status: list running/pending jobs for the user
2. Cancel any stale or unwanted jobs if approaching the 2-job limit
3. Verify resource availability on the interactive partition

### Training Code Testing Guidelines

When testing training code:

1. **Monitor loss convergence** - Verify the loss is decreasing as expected
2. **Track all metrics** - Watch training metrics, validation metrics, and any custom metrics
3. **Early termination is acceptable** - You do not need to wait for the full job to complete
   - If loss is trending down correctly
   - If metrics look healthy
   - If no errors or warnings appear
   - Then you can cancel the job early and report success
4. **Watch for warning signs**:
   - Loss not decreasing or exploding
   - NaN/Inf values in metrics
   - OOM errors
   - Unexpected crashes

### Job Management Workflow

```
1. List current jobs → Check if at limit
2. Cancel stale jobs → Free up slots if needed
3. Submit new job → With appropriate resources
4. Monitor output → Watch logs and metrics
5. Verify correctness → Loss going down, metrics healthy
6. Cancel early if validated → Don't waste cluster time
```

### Example Job Parameters

For interactive development:
- `partition=interactive`
- `time_limit="4:00:00"` (maximum allowed)
- `account="nvr_lpr_agentic"` (required for all jobs)
- Use appropriate GPU count for the model size

### Project Directory Organization

For each project, create a dedicated project directory under these slurm directories:
- `logs/` - For job output and error logs
- `results/` - For experiment results and outputs
- `Projects/` - For project source code and configurations

This keeps work organized and prevents file collisions between different projects.

### Cleanup Responsibilities

- Always cancel jobs that are no longer needed
- Check for orphaned jobs at the start of new sessions
- Do not leave jobs running unnecessarily after validation

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
