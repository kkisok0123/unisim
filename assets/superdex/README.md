# Local SuperDex assets

See the [SuperDex guide](../../docs/superdex.md) for installation, API usage,
supported inputs, comparison/viewer commands, and the maintained limitations report.

Asset payloads and their licenses stay local and are excluded from distributions.
The tracked [JSON inventory](../../docs/superdex-assets-inventory.json) records
provenance, dependency closure and integrity hashes; a candidate is not a runtime qualification.
Refresh it after intentional bundle changes with:

```bash
uv run --no-sync scripts/copy_superdex_assets.py --inventory-only
```
