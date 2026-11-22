"""Opt-in local reference additions from independently observed complete art.

The production encoder and 1..856 loader are unchanged. These references are
artwork evidence, never current-frame identity or input authorization.
"""

import json
import math
import hashlib
import unicodedata

import numpy as np
from PIL import Image

try:
    from agent.card_selection.embedding import EmbeddingGallery, EmbeddingCardRecognizer, extract_upgrade_marker
except ModuleNotFoundError:
    from card_selection.embedding import EmbeddingGallery, EmbeddingCardRecognizer, extract_upgrade_marker

from .masked_art import FrozenMaskedArt, PreparedMaskedArt
from .partial_card import LivePartialCardView, preprocess_live_card_view

PROFILE = 'hif-strong-reference-trial-v1'
VERSION = 'hif-trial-art-delta-v1'
FAMILIES = frozenset((('859', '860'), ('861', '862')))
# Existing families exposed by the current five-card/gray-layer observation.
# They retain their original groups and marker references, unlike new cards.
EXISTING_FAMILIES = frozenset((('24', '25'), ('146', '147'), ('596', '597')))
ART_FAMILIES = FAMILIES | EXISTING_FAMILIES
_EVIDENCE_KEYS = ('frame_id', 'source_record', 'card_id', 'title', 'title_confidence',
                  'upgrade_count', 'upgrade_confidence', 'identity_source', 'reviewer',
                  'title_reviewed', 'upgrade_reviewed', 'raw_ocr')


def _text(value):
    return ''.join(unicodedata.normalize('NFKC', value).split())


def _family_evidence(row, catalog):
    family = tuple(row['family'])
    if family not in ART_FAMILIES:
        raise ValueError('trial art family is outside the explicit whitelist')
    entries = [catalog.entries.get(cid) for cid in family]
    if (any(entry is None for entry in entries)
            or [entry.upgraded for entry in entries] != [False, True]
            or entries[0].plan != 'sense' or entries[1].plan != 'sense'
            or _text(entries[1].name) != _text(entries[0].name) + '+'):
        raise ValueError('trial art family has no confirmed catalog pair')
    observations = row['observations']
    if (not isinstance(observations, list) or len(observations) != 3
            or len({item['frame_id'] for item in observations}) != 3):
        raise ValueError('trial art requires three independently named source frames')
    proof = []
    for item in observations:
        if item.get('identity_source') == 'assistant_visual_family_review':
            # Complete current art can prove an old family even when the
            # title suffix and visible marker do not prove one upgrade state.
            if (family not in EXISTING_FAMILIES
                    or any(not isinstance(item.get(key), str) or not item[key]
                           for key in ('frame_id', 'source_record', 'title', 'reviewer'))
                    or item.get('card_id') is not None or item.get('upgrade_count') is not None
                    or item.get('title_confidence') is not None or item.get('upgrade_confidence') is not None
                    or item.get('title_reviewed') is not True or item.get('upgrade_reviewed') is not False
                    or not isinstance(item.get('raw_ocr'), dict)
                    or sum(_text(item['title']) == _text(entry.name) for entry in entries) != 1):
                raise ValueError('trial art family-only annotation is incomplete or resolves an upgrade')
            proof.append({key: item[key] for key in _EVIDENCE_KEYS if key in item})
            continue
        if (any(not isinstance(item.get(key), str) or not item[key]
                for key in ('frame_id', 'source_record', 'card_id', 'title'))
                or item['card_id'] not in family
                or type(item.get('upgrade_count')) is not int
                or item['upgrade_count'] not in (0, 1)):
            raise ValueError('trial art source identity is incomplete')
        entry = entries[item['upgrade_count']]
        if item['card_id'] != entry.card_id or _text(item['title']) != _text(entry.name):
            raise ValueError('trial art title and upgrade marker contradict its catalog ID')
        identity_source = item.get('identity_source', 'ocr')
        if identity_source == 'assistant_visual_review':
            # Static reference annotation is distinct from an automatic live
            # observation. Preserve weak/raw OCR rather than inventing a score.
            if (not isinstance(item.get('reviewer'), str) or not item['reviewer']
                    or item.get('title_reviewed') is not True
                    or item.get('upgrade_reviewed') is not True
                    or not isinstance(item.get('raw_ocr'), dict)
                    or item.get('title_confidence') is not None
                    or item.get('upgrade_confidence') is not None):
                raise ValueError('trial art visual annotation is incomplete')
        elif identity_source != 'ocr' or any(
                type(item.get(key)) not in (int, float) or not math.isfinite(item[key])
                or not .95 <= item[key] <= 1
                for key in ('title_confidence', 'upgrade_confidence')):
            raise ValueError('trial art title or upgrade evidence is unresolved')
        proof.append({key: item[key] for key in _EVIDENCE_KEYS if key in item})
    if len({(item['card_id'], _text(item['title']), item['upgrade_count']) for item in proof}) != 1:
        raise ValueError('trial art source identity changed between frames')
    return family, [entry.name for entry in entries], proof


def _source(recognizer, frozen):
    if (frozen.recognizer is not recognizer
            or set(recognizer.gallery.card_ids) != {str(i) for i in range(1, 857)}
            or frozen.exclusion_boxes != ((0., 0., .70, .40),)
            or not frozen._prepared):
        raise ValueError('trial art requires the complete original production references')
    return dict(model=recognizer.model_sha256, gallery=recognizer.gallery_sha256,
                classes=recognizer.classes_sha256, masked=frozen.gallery_sha256)


def _canonical_view():
    return LivePartialCardView(0, (0, 0, 180, 180), (0, 0, 180, 180),
                               (180, 180), 1., 'full', 'trial-reference')


def build_trial_art_delta(recognizer, frozen, *, revision, catalog, evidence, images):
    """Encode only confirmed complete square art; return a JSON-serializable delta.

    ``evidence`` contains family/observations rows. Each of three observations
    supplies the fields in _EVIDENCE_KEYS and a complete square ``art_box``.
    ``images`` maps those frame IDs to the original RGB/BGR source images in
    the recognizer's color order. Boxes are consumed here, never restored.
    """
    source = _source(recognizer, frozen)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError('trial art revision is required')
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= len(ART_FAMILIES):
        raise ValueError('trial art requires confirmed whitelisted families')
    keys = sorted(frozen._prepared)
    families, seen = [], set()
    for row in evidence:
        family, names, proof = _family_evidence(row, catalog)
        if family in seen:
            raise ValueError('duplicate trial art family')
        seen.add(family)
        variants = row.get('art_variants', [])
        if not isinstance(variants, list) or len(variants) > 1:
            raise ValueError('current trial permits at most one additional three-frame UI view')
        additional = [_family_evidence(dict(variant, family=list(family)), catalog)[2] for variant in variants]
        if any({item['card_id'] for item in group} != {proof[0]['card_id']} for group in additional):
            raise ValueError('trial art UI views disagree on the observed variant')
        full, plus_markers, masked = [], [], {f'{a}:{b}': [] for a, b in keys}
        if family in FAMILIES and any(item['upgrade_count'] != 1 for item in proof):
            raise ValueError('current trial additions require the actually observed plus variant')
        samples = [*row['observations'], *(item for variant in variants for item in variant['observations'])]
        for sample_index, item in enumerate(samples):
            image = np.asarray(images[item['frame_id']])
            box = item['art_box']
            if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
                    or not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(value) is not int for value in box)):
                raise ValueError('trial art source pixels or square ROI are invalid')
            x, y, width, height = box
            if not (width == height and width >= 120 and min(x, y) >= 0
                    and x+width <= image.shape[1] and y+height <= image.shape[0]):
                raise ValueError('trial art source must be a complete observed square')
            crop = image[y:y+height, x:x+width]
            if recognizer.embedder.source_color_order == 'BGR':
                crop = crop[:, :, ::-1]
            if sample_index < 3 and family in FAMILIES:
                plus_markers.append(extract_upgrade_marker(Image.fromarray(
                    np.ascontiguousarray(crop))).tolist())
            art = np.asarray(Image.fromarray(np.ascontiguousarray(crop)).resize(
                (180, 180), Image.Resampling.BILINEAR))
            view = _canonical_view()
            tensor = preprocess_live_card_view(art, view, source_color_order='RGB')
            full.append(recognizer.embedder.embed_tensor(tensor).tolist())
            for key in keys:
                tensor = preprocess_live_card_view(art, PreparedMaskedArt.common_view(view, key),
                    source_color_order='RGB', exclusion_boxes=frozen.exclusion_boxes)
                masked[f'{key[0]}:{key[1]}'].append(recognizer.embedder.embed_tensor(tensor).tolist())
        families.append(dict(family=list(family), names=names, observations=proof,
                             additional_observations=additional,
                             full=full, masked=masked,
                             plus_markers=plus_markers if family in FAMILIES else None))
    return dict(version=VERSION, revision=revision, source=source,
                mask_keys=[list(key) for key in keys], families=families)


class _TrialGallery(EmbeddingGallery):
    """Keep old marker resolution; never invent new normal/+ marker samples."""

    def infer_upgrade_state(self, image, box, hit, **kwargs):
        if set(hit.candidate_card_ids or (hit.card_id,)) <= self._old_ids:
            return self._old_gallery.infer_upgrade_state(image, box, hit, **kwargs)
        return None  # HIF must independently resolve the current visible marker.


class _OriginalMarkers:
    """Unchanged original marker indices; no manufactured new base template."""

    def __init__(self, source):
        self.source = source

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)) and not -len(self.source) <= index < len(self.source):
            raise ValueError('trial art new family has no normal/+ marker reference pair')
        return self.source[index]

    def __len__(self):
        return len(self.source)


def _append(old, rows, *, identity):
    vectors, names, ids, groups, upgrades = [], [], [], [], []
    for family, values in rows:
        existing = family in EXISTING_FAMILIES
        group = 'trial-art-' + '-'.join(family)
        if existing:
            groups_by_id = {cid: {gid for card_id,gid in zip(old.card_ids,old.visual_group_ids,strict=True)
                                  if card_id == cid} for cid in family}
            if any(len(value) != 1 for value in groups_by_id.values()):
                raise ValueError('existing trial family lacks unique original visual groups')
            observed_groups = set().union(*groups_by_id.values())
            if len(observed_groups) != 1:
                raise ValueError('existing trial family contradicts original visual groups')
            group = observed_groups.pop()
            if {cid for cid,gid in zip(old.card_ids,old.visual_group_ids,strict=True)
                    if gid == group} != set(family):
                raise ValueError('existing trial family does not cover its complete original group')
        values = np.asarray(values, dtype=np.float32)
        if (values.shape not in ((3, 128), (6, 128)) or not np.isfinite(values).all()
                or np.max(np.abs(np.linalg.norm(values, axis=1)-1)) > 1e-3):
            raise ValueError('trial art vectors must be complete normalized three-frame views')
        for index, vector in enumerate(values):
            for upgrade, cid in enumerate(family):
                vectors.append(vector)
                names.append(f'{cid}_trial-art-{index}')
                ids.append(cid)
                groups.append(group)
                upgrades.append(upgrade)
    gallery = _TrialGallery(np.concatenate((old.embeddings, vectors)),
        (*old.class_names, *names), (*old.card_ids, *ids),
        visual_group_ids=(*old.visual_group_ids, *groups),
        upgrade_counts=(*old.upgrade_counts, *upgrades),
        model_sha256=old.model_sha256, gallery_sha256=identity)
    # Constructor validation normalizes floats. Copy the already validated
    # original block back verbatim: this addition must not perturb old vectors.
    gallery.embeddings[:len(old.embeddings)] = old.embeddings
    gallery.upgrade_markers = None if old.upgrade_markers is None else _OriginalMarkers(old.upgrade_markers)
    gallery._old_gallery = old
    gallery._old_ids = frozenset(old.card_ids)
    return gallery


def attach_trial_art_delta(recognizer, frozen, payload, *, profile, catalog):
    """Attach one validated local delta before capture; production load stays fixed."""
    if profile != PROFILE:
        raise ValueError('trial art is restricted to the explicit strong-reference experiment')
    if (payload.get('version') != VERSION or not isinstance(payload.get('revision'), str)
            or not payload['revision'].strip() or payload.get('source') != _source(recognizer, frozen)
            or payload.get('mask_keys') != [list(key) for key in sorted(frozen._prepared)]):
        raise ValueError('trial art version or original reference identities differ')
    families = payload.get('families')
    if not isinstance(families, list) or not 1 <= len(families) <= len(ART_FAMILIES):
        raise ValueError('trial art family coverage is invalid')
    rows, seen = [], set()
    for row in families:
        family, names, _ = _family_evidence(row, catalog)
        if (family in seen or row.get('names') != names
                or set(row.get('masked', {})) != {f'{a}:{b}' for a, b in frozen._prepared}):
            raise ValueError('trial art family or mask coverage differs')
        additional = row.get('additional_observations', [])
        if not isinstance(additional, list) or len(additional) > 1:
            raise ValueError('trial art UI source coverage is invalid')
        for observations in additional:
            _, _, proof = _family_evidence(dict(family=list(family), observations=observations), catalog)
            if {item['card_id'] for item in proof} != {row['observations'][0]['card_id']}:
                raise ValueError('trial art UI sources disagree')
        samples = [*row['observations'], *(item for group in additional for item in group)]
        if (len({item['frame_id'] for item in samples}) != len(samples)
                or any(np.asarray(values).shape != (len(samples), 128)
                       for values in [row['full'], *row['masked'].values()])):
            raise ValueError('trial art vectors differ from complete independent source coverage')
        seen.add(family)
        if family in EXISTING_FAMILIES:
            if row.get('plus_markers') is not None:
                raise ValueError('existing trial family cannot replace original markers')
        else:
            markers = np.asarray(row.get('plus_markers'), dtype=np.float32)
            if (markers.shape != (3, 24, 24, 3) or not np.isfinite(markers).all()
                    or markers.min() < 0 or markers.max() > 1
                    or any(item['upgrade_count'] != 1 for item in row['observations'])):
                raise ValueError('trial art plus references are incomplete or not observed')
        rows.append((family, row))
    # Existing audit fields describe gallery/class bytes; they must distinguish
    # this local reference addition from the unchanged production artifacts.
    identity = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest().upper()
    def adapt(old_recognizer, vectors):
        return EmbeddingCardRecognizer(recognizer.embedder,
            _append(old_recognizer.gallery, vectors, identity=identity),
            classes_sha256=identity, upgrade_maximum_distance=recognizer.upgrade_maximum_distance,
            upgrade_minimum_margin=recognizer.upgrade_minimum_margin)
    result = adapt(recognizer, [(family, row['full']) for family, row in rows])
    prepared = {key: adapt(old, [(family, row['masked'][f'{key[0]}:{key[1]}'])
                               for family, row in rows])
                for key, old in frozen._prepared.items()}
    result.gkh_hif_trial_existing_art_families = tuple(family for family,row in rows
        if family in EXISTING_FAMILIES)
    result.gkh_hif_trial_inventory = dict(profile=PROFILE, revision=payload['revision'],
        families={'trial-art-'+'-'.join(family): dict(family=family,
            plus_markers=np.asarray(row['plus_markers'], dtype=np.float32))
                  for family, row in rows if family in FAMILIES})
    return result, FrozenMaskedArt(result, prepared, frozen.exclusion_boxes, identity)
