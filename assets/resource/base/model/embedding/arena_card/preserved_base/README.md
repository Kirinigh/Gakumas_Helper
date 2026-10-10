# Arena custom-card embedding

This directory contains the arena-only customized-card visual-family model. It
reuses the base card model's 128-dimensional embedding and Top-5 contract, but was trained
with class-independent customization overlays plus one locally retained positive
sample. The source screenshot is not committed.

The runtime acceptance threshold and minimum margin intentionally remain `null`.
Callers must fail closed until calibration freezes enough real arena positives and
negatives to approve them. Even after acceptance, this model identifies only the
visual family; the exact normal/+ business ID and `customizationId` must be
confirmed from the clicked card detail and the bundled `gakumas-data` catalog.

The removed `gakumas-tools` Softmax classifier is not a fallback for this model.

## 2026-09-21: authorized full-UI gallery augmentation

The encoder is unchanged. The gallery now preserves all 4180 existing rows and
adds the frozen 4176 complete-UI reference vectors (8356 rows, 880 business IDs).
Only this arena gallery changes; the base gallery remains unchanged. Membership
matches the frozen arena-only experiment, without identity-specific exceptions.
The new 877-880 rows retain their prior behavior.

The user authorized enablement before follow-up validation completes real negative rejection,
full-flow timing (P50/P95 <= 102% of baseline), and opponent validation. These gates
are PENDING, not passed. Candidate recall does not imply reduced clicks. Thresholds,
Top-5, three-frame and clicked-detail protections are unchanged.

Rebuild with tools/card-embedding/export_arena_full_ui_gallery.py using the frozen
report and original gallery baseline. Future raw gallery extensions must start
from the pre-union baseline, then append the frozen UI references. Roll back the
gallery, class table and manifest together to the pre-augmentation release if required.
