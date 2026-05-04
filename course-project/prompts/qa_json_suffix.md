
After completing your review, you MUST end your response with a fenced JSON block (nothing after it) in exactly this format:

```json
{
  "verdict": "APPROVED",
  "score": 0.9,
  "issues": [],
  "suggestions": ["Add type hints to public functions"],
  "tests_run": 5,
  "tests_passed": 5
}
```

The JSON block must be the LAST thing in your response. `verdict` must be exactly `"APPROVED"` or `"REVISION_NEEDED"`. `score` must be between 0.0 and 1.0.
