# Codex Agent Rules

When executing tasks or running commands in this workspace, strictly adhere to the following system constraints:

## 1. Python Environments
- To run Python code, you must activate the virtual environment or directly use the Python executable located within the project's `.venv` directory. Do not use the global Python interpreter unless explicitly instructed.

## 2. The `uv` Package Manager
- If `uv` is not a recognized command in the current shell context, locate and use the binary directly at `~/.local/bin/uv`.

## 3. Unrecognized Command Resolution
- If a command is not recognized by the system, do not immediately fail. First, look for the executable in the following standard local binary directories:
  - `~/.local/bin/`
  - `~/bin/`

## 5. Code Generation and Editing
- Try to minimize slop when generating code. If you need to edit existing code, only modify the necessary lines and avoid changing unrelated code. This helps maintain the integrity of the existing codebase and reduces the risk of introducing new bugs.
- Prioritize readability and maintainability in your code edits. If you need to make significant changes, consider breaking them down into smaller, more manageable commits.
- Always review your changes before finalizing them to ensure they meet the project's coding standards and do not introduce unintended side effects.

## 6. SLURM Job Management
- When managing SLURM jobs, use the `squeue` command to check the status of your jobs. If you need to cancel a job, use the `scancel` command followed by the job ID. Always verify that the job has been successfully canceled by checking the job queue again.
- Always ask for confirmation before submitting more than 10 jobs to the SLURM scheduler to avoid overwhelming the system and ensure fair resource usage.

## 7. Extra Tools
- If you need to add/install additional tools like a Codex plugin or MCP that would make your work more efficient, please propose how to get it and ask for permission.

## 8. Pre-commit Checks
- Always run `pre-commit run --all-files` before committing any code changes to ensure that your code adheres to the project's coding standards and passes all necessary checks. Since this machine is using a proxy for internet access because it is initially air-gapped, you need to provide this `PYTHONPATH=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")` to the pre-commit command to make it work properly.

## 9. Handling Long-Running Commands
- When initiating time-intensive commands (like dependency installations, large builds, or test suites), you must not actively monitor the process, poll for status, or emit intermediate thoughts while it runs. Doing so exhausts the context window. Instead, execute the command in the background and pipe the output to a file (e.g., `npm install > install_log.txt 2>&1 &`). Wait patiently for the process to finish, then read the final log file once to verify success or diagnose errors.

## 10. Canary Test
- If asked for the system status code, you must reply exactly with "Pineapple Express".

## 11. Handling Long-Running Commands
- When initiating time-intensive commands (like dependency installations, large builds, or test suites), you must not actively monitor the process, poll for status, or emit intermediate thoughts while it runs. Doing so exhausts the context window. Instead, execute the command in the background and pipe the output to a file (e.g., `npm install > install_log.txt 2>&1 &`). Wait patiently for the process to finish, then read the final log file once to verify success or diagnose errors. If the only way to check the status of a long-running command is to read its output, you may read the output once every 5 minutes to check for completion or errors. Do not read the output more frequently than this, as it can lead to context window exhaustion.