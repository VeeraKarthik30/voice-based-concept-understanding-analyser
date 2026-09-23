# VBCUA Upgrade Notes

This version of VBCUA was strengthened to address the main gaps identified in the project evaluation.

## Improvements applied

- Added safer rendering by escaping user-derived text before custom HTML display
- Added PDF content escaping for dynamic report fields
- Added audio upload validation for:
  - supported extensions
  - file signature matching
  - empty uploads
  - file size limits
- Added inactivity-based session expiry
- Enabled SQLite foreign-key enforcement and connection busy timeout
- Reduced in-memory analysis payload size by excluding raw waveform signal data from persisted result payloads
- Expanded the README with architecture, reliability, deployment, and limitations

## Tests added

- Authentication tests
- Database and foreign-key tests
- Upload helper and sanitization tests
- PDF generation test

## Verification result

All automated tests pass:

```bash
python3 -m pytest -q
```

Expected result:

```text
15 passed
```
