# Card embedding trainer

This toolchain keeps the approved 128-dimensional L2-normalised embedding and
Top-5 cosine nearest-neighbour retrieval route. It does not modify or replace the
production card classifier, UI, defaults, or click path.

`gk-img@4185529` is retained as historical evidence only and is rejected as a
training source. New training requires an authoritative `dataset_manifest.json`
whose validation state is `CLEAR`; the trainer verifies the exact ID 1..856
coverage, authoritative art/icon SHA-256 values, and decodability before importing
PyTorch or starting an epoch.

The current local dataset builder uses the pinned `gakumas-tools` numeric catalog,
the decoded master mapping, official DMM asset manifest/object CDN, and a pinned
Hatsuboshi Library renderer. Raw official game art, extracted icons, source dataset
manifests, checkpoints, augmentation outputs and evaluation captures remain under
`.local/` and must not be committed or redistributed. The separately inventoried
runtime model, numeric embedding gallery, class table and small upgrade-marker
reference tensors are the only publication outputs; see `ASSET_PROVENANCE.md` and
`THIRD_PARTY_NOTICES/` for their exact boundary and retained third-party rights.

```powershell
.local\task075-upstream\.venv\Scripts\python.exe tools\card-embedding\build_skill_card_dataset.py `
  --official-manifest .local\task075-upstream\manifest-pc.json `
  --official-manifest-label 705100:0036 `
  --object-manager-root .local\licensed\GkmasObjectManager\7da94255b10e367bf83fadb805a0c5f643998de9\GkmasObjectManager-7da94255b10e367bf83fadb805a0c5f643998de9 `
  --output-dir .local\task075-authoritative\pc-705100-0036
```

Embedding sees only official card artwork: score/cost text and contextual side
function badges are excluded. Normal/+ cards share one visual identity and the
upgrade marker is resolved by a separate ROI after retrieval. The 25 Legend cards
and singleton ID 23 need no upgrade state. The user-confirmed April-Fools artwork
reuse is resolved by both plan and upgrade state. Training uses:

```powershell
.local\task075-card-embedding\.venv\Scripts\python.exe tools\card-embedding\train_card_embedding.py `
  --art-root .local\task075-authoritative\pc-705100-0036\art `
  --icons-root .local\task075-authoritative\pc-705100-0036\icons `
  --dataset-manifest .local\task075-authoritative\pc-705100-0036\dataset_manifest.json `
  --output-dir .local\task075-card-embedding\authoritative-705100-0036 `
  --epochs 200 --device cuda
```

When the authoritative audit is `CLEAR`, the same command trains the fixed
embedding encoder, exports ONNX, builds the Top-5 gallery, and writes reproducible
manifests and deterministic augmentation metrics. Independent DMM atlas evaluation
remains a separate gate.
