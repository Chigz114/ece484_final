# Configuration

- `assets/manifest.json`: authoritative source, checkpoint, dataset, result, and
  SHA-256 inventory.
- `assets/code_locks/`: historical code-identity receipts for already generated
  datasets. These are evidence, not current runtime entry points.
- `environments/`: exact renderer/training environment overlays and pins.
- `hardware/preflight.template.json`: deliberately blocked hardware checklist.

Large artifacts referenced here are external and ignored by Git.
