# TASK-077 arena custom-card embedding

This directory contains the arena-only customized-card visual-family model. It
reuses the TASK-075 128-dimensional embedding and Top-5 contract, but was trained
with class-independent customization overlays plus one locally retained positive
sample. The source screenshot is not committed.

The runtime acceptance threshold and minimum margin intentionally remain `null`.
Callers must fail closed until TASK-390 freezes enough real arena positives and
negatives to approve them. Even after acceptance, this model identifies only the
visual family; the exact normal/+ business ID and `customizationId` must be
confirmed from the clicked card detail and the bundled `gakumas-data` catalog.

The removed `gakumas-tools` Softmax classifier is not a fallback for this model.
