"""Current thumbnail proof for the explicitly attached October card references.

The source titles label reference art only. Each new thumbnail still needs
whole-art retrieval and its own complete visible plus; no base is inferred.
"""

import math

import cv2
import numpy as np
from PIL import Image

try:
    from agent.card_selection.embedding import UPGRADE_MARKER_BOX, extract_upgrade_marker
except ModuleNotFoundError:
    from card_selection.embedding import UPGRADE_MARKER_BOX, extract_upgrade_marker

from .hif_trial_art import PROFILE, FAMILIES


def _source_plus_boundary(source_rgb, normalized_box):
    """Prove the original connected cross is enclosed before 24px resampling."""
    if (not isinstance(source_rgb, np.ndarray) or source_rgb.ndim != 3
            or source_rgb.shape[2] != 3 or source_rgb.dtype != np.uint8):
        return None
    height,width = source_rgb.shape[:2]
    left,top,right,bottom = (round(v*dimension) for v,dimension in
        zip(UPGRADE_MARKER_BOX,(width,height,width,height),strict=True))
    if right <= left or bottom <= top:
        return None
    red,green,blue = source_rgb.astype(np.float32).transpose(2,0,1)/255
    mask = ((red>.85)&(red-green>.12)&(red-blue>.12)).astype(np.uint8)
    _,_,stats,_ = cv2.connectedComponentsWithStats(mask)
    x,y,w,h = normalized_box
    observed = (x,y,x+w,y+h)
    matches = []
    for sx,sy,sw,sh,_ in stats[1:]:
        if not (left < sx and top < sy and sx+sw < right and sy+sh < bottom):
            continue
        projected = ((sx-left)*24/(right-left),(sy-top)*24/(bottom-top),
                     (sx+sw-left)*24/(right-left),(sy+sh-top)*24/(bottom-top))
        if all(math.floor(value) <= edge <= math.ceil(value)
               for edge,value in zip(observed,projected,strict=True)):
            matches.append(dict(source_component_box=[int(sx),int(sy),int(sw),int(sh)],
                source_marker_box=[left,top,right-left,bottom-top],
                reason='complete_source_component_before_resize'))
    return matches[0] if len(matches)==1 else None


def visible_trial_plus(query, *, source_rgb=None):
    """Require the complete warm-colored cross, independent of family artwork."""
    if query.shape != (24, 24, 3) or not np.isfinite(query).all():
        return None
    red, green, blue = query.transpose(2, 0, 1)
    mask = ((red > .85) & (red-green > .12) & (red-blue > .12)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    matches = []
    for label, (x, y, width, height, area) in enumerate(stats[1:], 1):
        if not (2 <= x <= 7 and 1 <= y <= 6 and 15 <= width <= 21
                and 11 <= height <= 17 and x+width <= 24 and y+height <= 20
                and 100 <= area <= 180):
            continue
        cross = labels[y:y+height, x:x+width] == label
        row, column = cross.sum(1), cross.sum(0)
        horizontal, vertical = int(row.argmax()), int(column.argmax())
        if (row.max() < width*.9 or column.max() < height*.9
                or not 3 <= horizontal <= height-4 or not 4 <= vertical <= width-5
                or row[0] >= width*.6 or row[-1] >= width*.6
                or column[0] >= height*.7 or column[-1] >= height*.7):
            continue
        evidence = dict(box=[int(x), int(y), int(width), int(height)], area=int(area))
        if x+width == 24:
            boundary = _source_plus_boundary(source_rgb,(x,y,width,height))
            if boundary is None:
                continue
            evidence['original_boundary'] = boundary
        matches.append(evidence)
    return matches[0] if len(matches) == 1 else None


class TrialInventoryIdentity:
    def __init__(self, recognizer):
        metadata = recognizer.gkh_hif_trial_inventory
        if (metadata.get('profile') != PROFILE or not isinstance(metadata.get('revision'), str)
                or not metadata['revision'] or not isinstance(metadata.get('families'), dict)
                or not 1 <= len(metadata['families']) <= 2):
            raise ValueError('trial inventory metadata is not explicitly enabled')
        self.recognizer = recognizer
        self.revision = metadata['revision']
        self.families = {}
        for group, row in metadata['families'].items():
            family = tuple(row['family'])
            markers = np.asarray(row['plus_markers'], dtype=np.float32)
            if (family not in FAMILIES or group != 'trial-art-'+'-'.join(family)
                    or markers.shape != (3, 24, 24, 3) or not np.isfinite(markers).all()
                    or markers.min() < 0 or markers.max() > 1
                    or any(visible_trial_plus(marker) is None for marker in markers)):
                raise ValueError('trial inventory lacks three complete positive marker references')
            members = {recognizer.gallery.card_ids[i] for i, value in
                       enumerate(recognizer.gallery.visual_group_ids) if value == group}
            if members != set(family):
                raise ValueError('trial inventory family differs from attached art')
            self.families[group] = (family, markers.copy())

    def identify(self, array, box, *, plan):
        hits = self.recognizer.gallery.search(self.recognizer.embedder.embed(array, box), top_k=5)
        if not hits or hits[0].visual_group_id not in self.families:
            return None  # Every old family continues through the original reader.
        top = hits[0]
        family, markers = self.families[top.visual_group_id]
        similarity = top.similarity
        margin = similarity-hits[1].similarity if len(hits) > 1 else None
        evidence = dict(source='trial_art_and_current_positive_marker', revision=self.revision,
                        family=family, input_authorized=False, normal_variant_supported=False)
        result = dict(card_id=None, visual_group=top.visual_group_id,
                      reason='trial_inventory_art_unresolved', embedding_similarity=similarity,
                      embedding_margin=margin, reference_evidence=evidence)
        if (plan != 'sense' or tuple(top.candidate_card_ids) != family
                or not all(type(value) in (int, float) and math.isfinite(value)
                           for value in (similarity, margin))
                or not .85 <= similarity <= 1 or margin < .05):
            return result
        x, y, width, height = box
        crop = array[y:y+height, x:x+width]
        if self.recognizer.embedder.source_color_order == 'BGR':
            crop = crop[:, :, ::-1]
        query = extract_upgrade_marker(Image.fromarray(np.ascontiguousarray(crop)))
        glyph = visible_trial_plus(query, source_rgb=np.ascontiguousarray(crop))
        evidence['current_plus_glyph'] = glyph
        result['reason'] = 'trial_inventory_positive_marker_unresolved'
        if glyph is None:
            return result
        score = min(float(cv2.minMaxLoc(cv2.matchTemplate(
            query, marker[4:20, 4:20].copy(), cv2.TM_SQDIFF))[0])/(16*16*3)
                    for marker in markers)
        evidence['positive_marker_distance'] = score
        if not math.isfinite(score) or score > self.recognizer.upgrade_maximum_distance:
            return result
        result.update(card_id=family[1], reason='accepted_trial_art_and_current_plus')
        return result
