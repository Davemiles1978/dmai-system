# Scripts

## Deploy safety: `preflight.sh` + pre-push hook

After the 2026-06-26 production incident (undefined `@require_master_password`
decorator crashed every new worker on boot while the old worker kept serving
`/health` 200), we now require a preflight check before every push to `main`
(the Render-deployed branch).

### One-time setup (per clone)

```bash
bash scripts/install-hooks.sh
```

This copies `scripts/git-hooks/pre-push` into `.git/hooks/pre-push`.
Git does **not** version-control hooks — every developer must run this once
after cloning. CI cannot enforce it remotely; this is local protection only.

### Run the preflight manually

```bash
bash scripts/preflight.sh
```

Exit code 0 → safe to push. Any other code → DO NOT PUSH.

### What it checks

| # | Check | Catches |
|---|-------|---------|
| A | `py_compile dmai_core_complete.py` + `compileall components/` | Syntax errors |
| B | **Full `import dmai_core_complete`** in a subshell with stripped env | NameError / ImportError at module level (the 2026-06-26 bug) |
| C | AST scan for undefined `@decorator` references | The same bug class, even before import — fast and explicit |
| D | Asserts `len(app.url_map._rules) >= 100` | Routes silently failing to register |
| E | Flask `test_client()` hits 5 known-stable routes, asserts non-404 | Specific route registration regressions |

Check B is the most important. `py_compile` passes for `@nonexistent_decorator`
because decorators resolve at import time, not compile time. Check B
exercises the actual import path Render's gunicorn worker uses.

### Emergency bypass

```bash
git push --no-verify
```

Use sparingly. Logs an audit trail in your shell history.

### When the preflight fails

Read the printed traceback / `[FAIL]` lines. Common failures:

- **Check B traceback shows `NameError: name 'X' is not defined`** → you referenced a symbol that doesn't exist. Either define it, import it, or remove the reference.
- **Check C lists an undefined decorator** → same root cause, found faster.
- **Check D fails with low route count** → import succeeded but `@app.route` blocks raised exceptions silently. Look at the traceback in Check B (it'll usually show the underlying error).
- **Check E shows 404 on a stable route** → route registration regressed. Check recent edits to `dmai_core_complete.py`.
