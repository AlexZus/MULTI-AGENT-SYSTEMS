
After completing the implementation, you MUST end your response with a fenced JSON block (nothing after it) in exactly this format:

```json
{
  "summary": "Brief description of what was implemented",
  "files_created": ["main.py", "tests/test_main.py"],
  "dependencies_installed": [],
  "tests_passed": true,
  "notes": ""
}
```

The JSON block must be the LAST thing in your response. `files_created` must list all files you created or modified using plain relative paths (e.g. `src/main.py`, `tests/test_main.py`).
