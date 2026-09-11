# Card embedding trainer

This toolchain keeps the approved 128-dimensional L2-normalised embedding and
Top-5 cosine nearest-neighbour retrieval route. It does not modify or replace the
production card classifier, UI, defaults, or click path.

`gk-img@4185529` is retained as historical evidence only and is rejected as a
training source. New training requires an authoritative `dataset_manifest.json`
whose validation state is `CLEAR`; the trainer verifies the manifest's exact business-ID set
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

## Gallery-only updates

Do not retrain the encoder for an ordinary same-domain card batch. The
gallery-only builder carries inherited NPZ rows forward without re-encoding them,
verifies the extension manifest's exact (possibly sparse) business-ID set, and
encodes only the IDs listed by the extension. An extension replaces every class
of any business ID it lists. The old dataset manifest and crosswalk remain the
identity source, but inherited art/icon files do not need to be copied, linked, or
present.

First materialize the exact append/replace delta from frozen catalog, decoded
PCard, official PC manifest, raw-art inventory and metadata-free fixed icons:

```powershell
.venv\Scripts\python.exe tools\card-embedding\build_skill_card_extension_dataset.py `
  --catalog .local\task095-upstream\gakumas-tools\data\skill_cards.json `
  --pcard .local\task095-upstream\pcard.json `
  --official-manifest .local\task095-upstream\official-pc.json `
  --raw-art-inventory .local\task095-upstream\raw-art-inventory.json `
  --raw-art-root .local\task095-upstream\raw-art `
  --fixed-icon-root .local\task095-upstream\fixed-icons `
  --base-gallery-manifest assets\resource\base\model\classify\card_embedding\manifest.json `
  --base-dataset-manifest .local\task075-authoritative\pc-705100-0036\dataset_manifest.json `
  --dataset-revision pc-<version>-<catalog-revision> `
  --catalog-revision <40-character-commit> `
  --output-dir .local\task095-card-extension
```

Repeat `--base-dataset-manifest` for every already accepted extension. The
materializer rejects any undeclared old identity drift, missing source, cross-ID
collision, metadata-bearing PNG or incomplete normal/`+` mapping and emits only
the new/replaced local files plus a `CLEAR` manifest.

```powershell
.venv\Scripts\python.exe tools\card-embedding\build_card_embedding_gallery.py `
  --input-mode EXTENSION `
  --dataset-manifest .local\task095-card-extension\dataset_manifest.json `
  --base-dataset-manifest .local\task075-authoritative\pc-705100-0036\dataset_manifest.json `
  --base-component-root assets\resource\base\model\classify\card_embedding `
  --output-dir .local\task095-card-gallery\candidate `
  --source-domain OFFICIAL_RAW_CARD_ART `
  --base-source-domain OFFICIAL_RAW_CARD_ART `
  --cache-dir .local\task095-card-gallery-cache
```

The extension manifest uses `scope.business_ids`. For a later batch whose
accepted base already contains earlier extensions, repeat
`--base-dataset-manifest` in chronological order (original full manifest first,
then each accepted extension manifest). Those retained manifests and crosswalks
are sufficient; inherited image directories are still unnecessary.

Both source domains are mandatory. Cache identity includes the pinned model hash,
preprocessing contract, canonical pixel hash, and ONNX Runtime/provider
fingerprint. `FIXED_RENDERED_PNG` is a candidate-only domain and its output
manifest always reports `PRODUCTION_HANDOFF_BLOCKED`; it cannot be handed to a
production consumer without separate frozen-set equivalence evidence.

Fixed rendered icons are useful diagnostics and remain the authoritative source
for the separate normal/`+` marker templates. They are not, by themselves, the
card-art embedding acceptance set: the accepted card-art encoder was gated on
official raw-art augmentations and frozen DMM ROIs. A future batch therefore
triggers encoder retraining only when those gates show a regression or the visual
domain changes; missing real samples must remain explicitly `PENDING`.

Evaluate every candidate against both the accepted gallery and the accepted
historical dataset before promotion. The evaluator checks all inherited NPZ
arrays byte-for-byte, evaluates official raw art, compares fixed-rendered results
with the same-size historical tail, and keeps real DMM coverage explicit:

```powershell
.venv\Scripts\python.exe tools\card-embedding\evaluate_card_embedding_gallery_update.py `
  --candidate-root .local\task095-card-gallery\candidate `
  --accepted-component-root assets\resource\base\model\classify\card_embedding `
  --extension-manifest .local\task095-card-extension\dataset_manifest.json `
  --accepted-dataset-manifest .local\task075-authoritative\pc-705100-0036\dataset_manifest.json `
  --output .local\task095-card-gallery\evaluation.json
```

Promotion is a separate deterministic step. It binds the evaluation to the exact
candidate manifest, model, gallery and class table; requires inherited rows to be
bitwise equal, official raw-art Top-1/Top-5 to pass and the new visual-group margin
to be at least `0.01`. Fixed-rendered scores remain diagnostic. A missing real DMM
set produces `READY_WITH_REAL_SAMPLES_PENDING`, never a false `READY`:

```powershell
.venv\Scripts\python.exe tools\card-embedding\promote_card_embedding_gallery.py `
  --component-root .local\task095-card-gallery\candidate `
  --evaluation .local\task095-card-gallery\evaluation.json `
  --output-dir .local\task095-card-gallery\ready
```

When `real_dmm_samples.status` is `PASS`, promotion additionally requires the
same schema-v2 report through `--real-sample-report`. The promoter reads that
file itself and binds its SHA-256, model/gallery hashes, exact business-ID
coverage, successful sample rows and sample count. A `PENDING` evaluation must
omit this argument.

For arena consumers, run the same build/evaluate/promote sequence against the
arena component, then regenerate the compact badge reference with
`build_card_badge_gallery.py`. The accepted component becomes the next batch's
base, so routine same-domain updates encode only new or explicitly replaced IDs;
old image files and old encoder calls are not repeated.

Repeat `--accepted-dataset-manifest` in the evaluator in the same chronological
order used by the gallery builder. Unlike the gallery build itself, the
fixed-rendered historical diagnostic reads the retained icon inventories, so
keep those small accepted icon sets under `.local/`; raw training art remains
unnecessary for inherited gallery rows.

## Live card-domain corrections for the compact arena gallery

A card that already retrieves the correct 128-dimensional visual family can
still be confused by the compact `arena_badge_reference` identity gallery after
runtime overlays are applied. This is a prototype-gallery correction, not an
encoder retrain. Use it only when an independently resolved detail title and
full effect bind one clean, user-authorized local card crop to its business ID.

Keep the screenshot and its private manifest under `.local/`. The manifest must
bind the image SHA-256, ROI, expected business ID and visual group, declare raw
publication `EXCLUDED`, and list the target, sibling and hard-negative IDs for
evaluation. The builder validates those declarations and the derived image
bytes; it does not independently parse a detail report. Therefore the label
must come from an explicit user-approved independent detail result, or from a
separately bound private evidence report when no such approval exists. The
builder publishes only a derived 8x16 BGR `top_coarse` row;
`green_mask` is copied from the existing clean row in the same visual group.
All inherited NPZ rows remain byte-identical, and accepted public batch lineage
is accumulated in `source.live_reference_updates`.

```powershell
.venv\Scripts\python.exe tools\card-embedding\build_card_badge_gallery.py `
  --base-component-root assets\resource\base\model\embedding\arena_badge_reference `
  --arena-component-root assets\resource\base\model\embedding\arena_card `
  --live-reference-manifest .local\card-live-update\live_reference_manifest.json `
  --output-dir .local\card-live-update\candidate
```

Build twice into separate directories and require identical gallery and manifest
bytes. A live build is always `PENDING_VALIDATION` and has no evaluation binding.
Then evaluate against the accepted base, the private live manifest and a
frozen authoritative card dataset. Hard negatives are evaluation-only guards:
they are not relabelled or appended as target rows. The evaluator replays every
historical compact row, rejects cross-group exact collisions, checks bounded ROI
geometry, requires exact coverage of every crosswalk-declared hard-negative asset
variant, writes only aggregate confusion evidence, and can atomically bind the
report into a promoted component:

```powershell
.venv\Scripts\python.exe tools\card-embedding\evaluate_card_badge_gallery_update.py `
  --base-component-root assets\resource\base\model\embedding\arena_badge_reference `
  --candidate-component-root .local\card-live-update\candidate `
  --live-reference-manifest .local\card-live-update\live_reference_manifest.json `
  --card-dataset-manifest .local\authoritative-card-dataset\dataset_manifest.json `
  --output-report .local\card-live-update\evaluation.json `
  --promoted-output-dir .local\card-live-update\promoted
```

The report never contains local paths, ROI coordinates, screenshots, private
sample hashes or the private manifest hash. A training crop is not an independent
live holdout: until a different authorized scene passes, production handoff stays
`READY_WITH_REAL_SAMPLES_PENDING` and the report states `PENDING` explicitly.
Only a promoted component may be used as the base of another live append; the
builder rejects unpromoted or report-mismatched live lineage. The release builder
also reopens the bound evaluator-v1 report and verifies its hash, complete gate
set, actual `L####` row-to-batch identities, confusion counts, candidate gallery
hash/row count and holdout status. This public-safe aggregate report is part of
the promoted component and release package. The private manifest, source crop and
maintenance evaluator are not packaged, and the runtime never executes the report.

Each public lineage row also retains the hard-negative business IDs declared for
that revision. Every later live evaluation unions those IDs and replays every
authoritative asset variant for the cumulative set, so a new report cannot forget
an older confusion guard. Evaluator v1 deliberately supports only `PENDING`
independent holdout evidence and `READY_WITH_REAL_SAMPLES_PENDING` promotion; a
real independent-holdout PASS requires a higher report schema that binds the
actual holdout evidence.

Historical live prototype rows are replayed exactly, but evaluator v1 does not
reopen a different older target's private crop and 41 geometry variants. A later
sample for the same target can be appended through the accepted base; before a
second distinct live target is introduced, add cumulative private-target replay
and its fail-closed regression. Keep every accepted private crop and manifest
under `.local/`; the public aggregate cannot reconstruct them.

Evaluator v1 also treats cumulative hard-negative IDs as protected identities.
If an older hard-negative ID later needs to become a live correction target, add
an edge-aware schema-v2 migration and regression suite first; do not delete that
ID from historical lineage or edit the handoff state. This restriction prevents
continuous live appends from silently dropping an earlier cross-ID guard.

A later fixed official-card extension built directly from a production badge
that already has live lineage preserves that lineage and is deliberately emitted
as `PENDING_VALIDATION`, because the earlier report no longer binds the changed
gallery. The routine path is to build the extension from the latest accepted
lineage-free official badge baseline, then replay and re-evaluate every retained
private live manifest in revision order. This route is regression-tested against
a changed official base and currently has one retained correction (529). Do not
drop lineage, lose its private replay input, or edit READY by hand.

## Upgrade-count glyph OCR evidence

`evaluate_upgrade_count_glyph_ocr.py` is a maintenance-only evaluator for the
small customization-count glyph. It does not change the runtime and does not
produce a model or gallery. The evaluator accepts whole arena-reader reports as
separate development and holdout inputs. The two splits must not share an exact
normalized OCR input after q4 binarization and tight cropping. A report must be
successful and carry the generator-emitted 40-character source revision. Truth
must join the same group index across `member.groups` and
`member.evidence.groups`, then come from one matching `positive_cards` record
with an unconstrained, twice-confirmed, unique positive detail result and
`expected_count=null`; final identity, customizations, detail check, and glyph
diagnostic must agree. Legacy reports without that explicit positive provenance
fail closed even when they contain a stable descriptor. A source revision added
by hand is not evidence: the report generator must emit and version that field.
Slots that contributed auxiliary glyph/OCR, badge-constrained, generic-cost, or
card-face evidence are rejected, and three exact descriptor-and-geometry frames
are required.

The frozen OCR view contract is 12x18 q4 `>0`, tight crop, 8/16/24-cell quiet
zones, 48-pixel height with cubic interpolation, and both polarities. A digit is
accepted only when the six recognition-only views contain no other digit and at
least two complete normal/inverted pairs agree. Pass the expected SHA-256 of
`rec.onnx`, `keys.txt`, and the frozen skill-card/customization catalogs on every
run; dependency drift fails closed.

Write the result below `.local/`. Output is aggregate-only: it contains counts,
reject/wrong totals, collision totals, public dependency hashes, and catalog
reachability, but never input paths or names, card IDs, member/group/card slots,
descriptors, screenshots, or private sample hashes. Missing authoritative
digits remain domain gaps. The legal recognition domain remains 1-9, but this
maintenance batch evaluates coverage only for 3-9 because 1/2 are owned by the
preceding independently validated arena-reader evidence. Only a reject or wrong result in the independent
3-9 holdout can justify proposing a new artifact; development failures and 1/2
diagnostics cannot. Cross-label collisions in either split or overall fail
before OCR. The current report generator does not yet emit the required source
revision or unconstrained positive record, so its existing reports correctly
produce zero authoritative samples. Digits taken from cost, stamina, parameter, or other
carriers may be used only as stress tests and must not open the customization
glyph acceptance gate.
# 全量技能卡分类索引

`assets/data/skill_card_classification.json` 以 `business_id` 连接三套图库的全部图像变体，保存每张卡的计划、种类、稀有度、普通／＋状态及源字段。索引支持同维度多选取并集、跨维度取交集；空选返回空集，未知标签报错。`select_ids(index, plan=["free", "sense"], rarity="L")` 可用于离线查询。计划筛选不会自动添加共通卡，调用者须显式选择。

- 计划：`free/sense/logic/anomaly`，使用官方主表并核对目录。
- 稀有度：`L/SSR/SR/R/N`。眠気原目录的 `T` 保留在源字段，分类按官方主表归 `N`。
- 种类优先级：全部 `L` → `other`；其余按官方固有来源或目录明确的 `pIdol/support` → `p_idol/support`；剩余名称含「基本」→ `basic`；其余 → `other`。官方关联的固有传说卡同样归 `other`，保留归属源字段但不据此划为固有种类。
- 源数据保留官方计划、稀有度、四类归属字段，以及目录 sourceType、pIdolId、type、rarity 和分类依据。目录固有来源可补足官方归属空值；不从图片外观或 pIdolId 单独推断固有种类。
- 构建会核对冻结来源、已接受的 ID 映射、三套图库的实际 ID 集合和普通／＋标签一致性。官方名称别名使用已接受 crosswalk。

复建当前 876 张卡索引（在项目根目录，使用项目虚拟环境 Python；下列输入路径须替换为已核验的冻结来源）：

```powershell
python tools/card-embedding/build_skill_card_classification.py `
  --catalog .local/classification-inputs/skill_cards.json `
  --master .local/classification-inputs/catalog-pcard.json `
  --dataset-manifest .local/classification-inputs/base/dataset_manifest.json `
  --dataset-manifest .local/classification-inputs/extension-1/dataset_manifest.json `
  --dataset-manifest .local/classification-inputs/extension-2/dataset_manifest.json `
  --dataset-manifest .local/classification-inputs/extension-3/dataset_manifest.json `
  --dataset-manifest .local/classification-inputs/extension-4/dataset_manifest.json `
  --component-root assets/resource/base/model/classify/card_embedding `
  --component-root assets/resource/base/model/embedding/arena_card `
  --component-root assets/resource/base/model/embedding/arena_badge_reference `
  --catalog-revision 0d0a85145258f700dbbab66d31acf87361297a37 `
  --revision skill-card-classification-876-v1 `
  --output assets/data/skill_card_classification.json
```

冻结输入不随公开源码分发，独立检出需另行取得同一来源文件。新增卡时须重新构建并验证覆盖，不能只修改索引计数。竞技场现已消费分类元数据，按舞台计划与卡位种类前置筛选技能卡候选；这不改变模型或图库向量，也不证明实机识别率改善。
