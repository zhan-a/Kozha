# Parser fixtures

25 hand-written prose descriptions that exercise the
`backend.chat2hamnosys.parser.description_parser.parse_description` pipeline.

Each fixture is a single JSON file with the shape:

```jsonc
{
  "id": "01-well-specified-hello-wave",
  "category": "well_specified",
  "prose": "...",                            // the user-facing description
  "expected_populated": [...],               // slot names that must be non-null
  "expected_gap_fields": [...],              // slot names that must appear in gaps[]
  "recorded_response": { ... },              // raw JSON object the LLM returned
  "notes": "..."                             // optional, free-form
}
```

`recorded_response` is stored as a parsed JSON object (not a string) for
readability; the test harness re-serialises it before feeding it to
`_build_parse_result`.

## Categories

| Category               | Count |
|------------------------|-------|
| `well_specified`       | 5     |
| `one_missing`          | 5     |
| `ambiguous_handshape`  | 4     |
| `regional_variant`     | 4     |
| `non_manuals`          | 4     |
| `register`             | 3     |
| **Total**              | **25**|

## How these recordings were produced

The first recording pass was hand-crafted to match what `gpt-4o` should
produce given the system prompt in `description_parser.SYSTEM_PROMPT`.
Recordings are intended to be refreshed once the author runs the
`record_fixtures.py` script against the real API — at that point the
responses will be overwritten with genuine LLM output and the eval numbers
in `docs/chat2hamnosys/05-description-parser-eval.md` should be regenerated.

When a recording is refreshed, the `expected_populated` /
`expected_gap_fields` lists should NOT be updated blindly to match the new
response — they represent the oracle (what the parser *should* return).
If the LLM fails to meet them, the eval doc records that as an accuracy
miss, not a fixture bug.
