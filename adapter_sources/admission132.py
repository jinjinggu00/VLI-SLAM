import math

def approve(bindings, records, cfg):
    """Admission adapter only; frozen pixel/3D thresholds and existing factors."""
    pairs = bindings.get('loop_pairs', [])
    ids = [(int(p['i']), int(p['j'])) for p in pairs]
    evidence = {(int(r['i']), int(r['j'])): r for r in records}
    if not pairs or len(set(ids)) != len(ids) or len(evidence) != len(records) or set(evidence) != set(ids):
        raise ValueError('Missing, duplicated or unknown final pair evidence')
    features = {k: [] for k in ids}
    seen = set()
    for p in bindings.get('points', []):
        key = tuple(map(int, p['pair']))
        if key not in features or len(p['support']) != 2:
            raise ValueError('Unknown original point support')
        a, b = p['support']
        if (int(a[0]), int(b[0])) != key:
            raise ValueError('Original point support direction differs')
        ia, ib = int(a[1]), int(b[1])
        if min(ia, ib) < 0 or ia % 5 == 0 or ib % 5 == 0:
            raise ValueError('Held or invalid identity in fitted original support')
        for node, feature in (a, b):
            identity = (int(node), int(feature))
            if identity in seen:
                raise ValueError('Duplicated original fitted identity')
            seen.add(identity)
        features[key].append([ia, ib])
    approved, rejected = [], []
    for p, key in zip(pairs, ids):
        r = evidence[key]
        names = ('after_median_px', 'after_reverse_median_px', 'after_p90_px', 'depth_median_m', 'depth_p90_m')
        if r.get('n', 0) < 5 or not features[key] or any(k not in r or not math.isfinite(float(r[k])) or r[k] < 0 for k in names):
            raise ValueError('Invalid or absent final measurement evidence')
        failures = []
        if max(r['after_median_px'], r['after_reverse_median_px']) > cfg['held_median_px'] or r['after_p90_px'] > cfg['held_p90_px']:
            failures.append('post_global_held_reprojection')
        if r['depth_median_m'] > cfg['depth_median_m'] or r['depth_p90_m'] > cfg['depth_p90_m']:
            failures.append('post_global_held_depth_3d')
        if sorted(r.get('failures', [])) != sorted(failures):
            raise ValueError('Recorded final measurement classification disagrees with frozen thresholds')
        item = dict(i=key[0], j=key[1], kind=p['kind'], evidence=r, feature_pairs=features[key])
        if failures:
            if p['kind'] != 'loop':
                raise ValueError('A failed temporal observation blocks release')
            item['reason'] = 'unchanged_post_global_geometry_gate'
            rejected.append(item)
        elif p['kind'] in ('loop', 'temporal'):
            approved.append(item)
        else:
            raise ValueError('Unknown original factor kind')
    return dict(approved=approved, rejected=rejected, frozen_thresholds=cfg,
                independent_validation_claimed=False, frame_coverage_unchanged=True,
                scope='Existing image/cloud landmark pairs only. Validation observations select loop admission; they are not independent accuracy evidence.')
