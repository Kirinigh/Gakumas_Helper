"""Pixel evidence for a zero delta, independent of any expected card effect."""


def corroborated_speck_zero(observations):
    """Two clear frames corroborate at most two interior pixels in the third.

    The existing red-word reader requires at least three pixels per component.
    Raw pixels remain recorded and are never relabelled as an OCR reading.
    The caller must additionally prove unchanged absolute resource values.
    """
    if not isinstance(observations, (tuple, list)) or len(observations) != 3:
        return False
    clear, speck = 0, 0
    region = None
    for row in observations:
        evidence = row.get('red_evidence') or {}
        roi = evidence.get('roi')
        if (row.get('accepted') is not False or row.get('value') is not None
                or evidence.get('accepted') is not False
                or evidence.get('bottom_guard_red_pixels') != 0
                or evidence.get('component_count') != 0 or evidence.get('components') != []
                or evidence.get('pixel_count') != 0 or evidence.get('box') is not None
                or not isinstance(roi, list) or len(roi) != 4
                or any(type(v) is not int for v in roi) or min(roi[2:]) <= 2
                or row.get('roi') != roi or row.get('reason') != evidence.get('reason')
                or region is not None and roi != region):
            return False
        region = roi
        count = evidence.get('raw_red_pixel_count')
        if type(count) is not int:
            return False
        if count == 0 and evidence.get('red_present') is False:
            if (evidence.get('reason') != 'resource_delta_red_evidence_absent'
                    or evidence.get('subcomponent_specks')):
                return False
            clear += 1
            continue
        points = evidence.get('subcomponent_specks')
        if (type(count) is not int or not 0 < count < 3
                or evidence.get('red_present') is not True
                or evidence.get('reason') != 'resource_delta_red_evidence_invalid'
                or not isinstance(points, list) or len(points) != count
                or any(not isinstance(p, list) or len(p) != 2
                    or any(type(v) is not int for v in p)
                    or not roi[0] < p[0] < roi[0]+roi[2]-1
                    or not roi[1] < p[1] < roi[1]+roi[3]-1 for p in points)
                or len({tuple(p) for p in points}) != count):
            return False
        speck += 1
    return clear == 2 and speck == 1
