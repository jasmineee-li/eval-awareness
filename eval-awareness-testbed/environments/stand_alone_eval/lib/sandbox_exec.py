from typing import Any

import inspect_ai.util

DEFAULT_SANDBOX_EXEC_TIMEOUT_SEC = 180  # approx 3 minutes, chosen arbitrarily


async def exec_with_timeout(
    sandbox_env: inspect_ai.util.SandboxEnvironment,
    cmd: list[str],
    input: str | bytes | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    timeout: int | None = None,
) -> inspect_ai.util.ExecResult[str]:
    """Execute a sandbox command with an enforced timeout.

    By default, inspect's `sandbox().exec(...)` command will accept running without a timeout, which will hang
    literally forever if the command hangs (for example, due to a docker failure).

    Forwards arguments to inspect's `sandbox().exec(...)` command.

    ```python
    result = await exec_with_timeout(sandbox(), ["ls", "-l", "/nonexistent"])

    # equivalent to:
    #
    result = await sandbox().exec(["ls", "-l", "/nonexistent"], timeout=DEFAULT_SANDBOX_EXEC_TIMEOUT_SEC)
    ```

    """
    timeout = timeout or DEFAULT_SANDBOX_EXEC_TIMEOUT_SEC
    env = env or dict()
    return await sandbox_env.exec(cmd, input, cwd, env, user, timeout=timeout)


async def checked_exec(sandbox: inspect_ai.util.SandboxEnvironment, cmd: list[str], **kwargs: Any) -> str:
    """Executes a command in a sandbox and raises an exception if it returns a nonzero exit code."""
    result = await exec_with_timeout(sandbox, cmd, **kwargs)
    if result.returncode != 0:
        raise Exception(f"Command: {cmd} returned {result.returncode}. Output:\n{result.stderr}\n{result.stdout}")
    return result.stdout
