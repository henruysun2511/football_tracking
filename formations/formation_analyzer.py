import numpy as np
from collections import Counter

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


KNOWN_FORMATIONS = {
    (4, 4, 2): "4-4-2",
    (4, 3, 3): "4-3-3",
    (4, 2, 3, 1): "4-2-3-1",
    (3, 5, 2): "3-5-2",
    (3, 4, 3): "3-4-3",
    (5, 3, 2): "5-3-2",
    (5, 4, 1): "5-4-1",
    (4, 1, 4, 1): "4-1-4-1",
    (4, 5, 1): "4-5-1",
    (4, 3, 2, 1): "4-3-2-1",
    (4, 1, 3, 2): "4-1-3-2",
    (3, 4, 1, 2): "3-4-1-2",
    (3, 6, 1): "3-6-1",
    (4, 2, 2, 2): "4-2-2-2",
    (4, 3, 1, 2): "4-3-1-2",
    (4, 1, 5): "4-1-5",
    (3, 3, 4): "3-3-4",
    (5, 2, 3): "5-2-3",
    (4, 2, 4): "4-2-4",
}


def _normalize_direction(positions, pitch_length):
    """Flip x so defenders are at low x, forwards at high x."""
    mean_x = np.mean([p[0] for p in positions])
    if mean_x > pitch_length / 2:
        return [(pitch_length - p[0], p[1]) for p in positions]
    return positions


def _classify_player(row, num_lines, line_width):
    """Classify a player's position index into a line (0=def, 1=mid, 2=fwd)."""
    return min(int(row * num_lines / line_width), num_lines - 1)


def detect_formation(positions, pitch_length=12000, method='kmeans'):
    """
    Detect formation from player positions on pitch.

    Args:
        positions: list of (x, y) tuples for 10 outfield players
        pitch_length: length of pitch in model units
        method: 'kmeans' or 'quantile'

    Returns:
        tuple: formation key e.g. (4, 3, 3), (4, 2, 3, 1), etc.
        str: formation name e.g. "4-3-3"
    """
    if len(positions) < 10:
        # Need all 10 outfield players
        return None, "unknown"

    # Normalize direction: defenders at low x, forwards at high x
    norm = _normalize_direction(positions, pitch_length)

    # Sort by x (defensive to offensive)
    sorted_pos = sorted(norm, key=lambda p: p[0])
    xs = np.array([p[0] for p in sorted_pos])

    if method == 'quantile':
        return _detect_quantile(xs, sorted_pos)
    else:
        return _detect_kmeans(xs, sorted_pos)


def _cluster_lines_kmeans(xs, n_clusters=3):
    """1D k-means clustering for line detection."""
    if _HAS_SKLEARN:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=5)
        labels = km.fit_predict(xs.reshape(-1, 1))
        return labels, km.cluster_centers_.flatten()
    return _cluster_lines_simple(xs, n_clusters)


def _cluster_lines_simple(xs, n_clusters=3):
    """Simple threshold-based clustering as k-means fallback."""
    sorted_xs = np.sort(xs)
    n = len(sorted_xs)

    if n_clusters == 3:
        # Split into 3 equal-ish groups
        d1 = n // 3
        d2 = 2 * n // 3
        thresh1 = (sorted_xs[d1 - 1] + sorted_xs[d1]) / 2 if d1 < n else sorted_xs[-1]
        thresh2 = (sorted_xs[d2 - 1] + sorted_xs[d2]) / 2 if d2 < n else sorted_xs[-1]
        labels = np.zeros(n, dtype=int)
        labels[xs > thresh2] = 2
        labels[(xs > thresh1) & (xs <= thresh2)] = 1
    elif n_clusters == 4:
        d1 = n // 4
        d2 = n // 2
        d3 = 3 * n // 4
        thresh1 = (sorted_xs[d1 - 1] + sorted_xs[d1]) / 2 if d1 < n else sorted_xs[-1]
        thresh2 = (sorted_xs[d2 - 1] + sorted_xs[d2]) / 2 if d2 < n else sorted_xs[-1]
        thresh3 = (sorted_xs[d3 - 1] + sorted_xs[d3]) / 2 if d3 < n else sorted_xs[-1]
        labels = np.zeros(n, dtype=int)
        labels[xs > thresh3] = 3
        labels[(xs > thresh2) & (xs <= thresh3)] = 2
        labels[(xs > thresh1) & (xs <= thresh2)] = 1
    else:
        labels = np.zeros(n, dtype=int)

    return labels, None


def _detect_kmeans(xs, sorted_pos):
    """Detect formation using k-means clustering into lines."""
    # Try 3-line formation first (def-mid-fwd)
    labels_3, centers_3 = _cluster_lines_kmeans(xs, 3)
    counts_3 = [int((labels_3 == i).sum()) for i in range(3)]
    # Sort by center position (def=0, mid=1, fwd=2)
    order_3 = np.argsort(centers_3) if centers_3 is not None else [0, 1, 2]
    def_cnt = counts_3[order_3[0]]
    mid_cnt = counts_3[order_3[1]]
    fwd_cnt = counts_3[order_3[2]]

    # Check if formation has 4 lines (like 4-2-3-1, 4-1-4-1)
    # The mid line split into 2 would have a clear gap
    labels_4, centers_4 = _cluster_lines_kmeans(xs, 4)
    counts_4 = [int((labels_4 == i).sum()) for i in range(4)]
    order_4 = np.argsort(centers_4) if centers_4 is not None else [0, 1, 2, 3]
    g1 = counts_4[order_4[0]]  # def
    g2 = counts_4[order_4[1]]  # def-mid or mid
    g3 = counts_4[order_4[2]]  # mid-fwd or mid
    g4 = counts_4[order_4[3]]  # fwd

    # Decide if 3-line or 4-line formation fits better
    # 4-line formations have 2 mids lines summing to ~4
    mid2_sum = g2 + g3
    if (min(g2, g3) >= 1 and 3 <= mid2_sum <= 5 and
            g1 + g2 + g3 + g4 == 10):
        formation = (g1, g2, g3, g4)
    else:
        formation = (def_cnt, mid_cnt, fwd_cnt)
        # If def+mid+fwd != 10, last player gets pushed to nearest line
        # This shouldn't happen with proper clustering

    formation = _verify_formation(formation)
    name = KNOWN_FORMATIONS.get(formation, "-".join(str(c) for c in formation))
    return formation, name


def _detect_quantile(xs, sorted_pos):
    """Detect formation using percentile-based approach."""
    total = len(xs)
    # Split into 3 lines using x position percentiles
    # Defenders: bottom 30-40% x, Midfielders: middle 30-40%, Forwards: top 30-40%
    # Use gaps between players to find natural splits

    # Find largest gaps between consecutive players (sorted by x)
    gaps = [(xs[i + 1] - xs[i], i) for i in range(total - 1)]
    gaps.sort(reverse=True)

    # Top 2 gaps define the 3 lines
    gap_indices = sorted([g[1] for g in gaps[:2]])

    def_cnt = gap_indices[0] + 1
    mid_cnt = gap_indices[1] - gap_indices[0]
    fwd_cnt = total - gap_indices[1] - 1

    formation = (def_cnt, mid_cnt, fwd_cnt)
    formation = _verify_formation(formation)
    name = KNOWN_FORMATIONS.get(formation, "-".join(str(c) for c in formation))
    return formation, name


def _verify_formation(formation):
    """Verify formation sums to 10 and adjust if needed."""
    formation = tuple(formation)
    s = sum(formation)
    if s != 10:
        # Distribute surplus/deficit to the most variable line (midfield)
        formation = list(formation)
        diff = 10 - s
        if len(formation) == 3:
            formation[1] += diff  # adjust midfield
        elif len(formation) == 4:
            formation[2] += diff  # adjust second midfield line
        formation = tuple(formation)

    # Clamp to valid ranges
    clamped = []
    for c in formation:
        clamped.append(max(1, min(6, c)))
    # Re-adjust sum
    s2 = sum(clamped)
    while s2 != 10:
        if s2 < 10:
            clamped[-1] += 1
        else:
            clamped[-1] -= 1
        s2 = sum(clamped)
    return tuple(clamped)


def detect_team_formation(tracks, team_id, frame_nums=None, method='kmeans'):
    """
    Detect formation for a team over specified frames.

    Args:
        tracks: tracks dict from the pipeline
        team_id: 1 or 2
        frame_nums: list of frame indices, or None for all
        method: 'kmeans' or 'quantile'

    Returns:
        (formation_tuple, name, conf_score)
    """
    if frame_nums is None:
        frame_nums = range(len(tracks['players']))

    detections = []
    for fn in frame_nums:
        frame_players = []
        for pid, data in tracks['players'][fn].items():
            if data.get('team') == team_id:
                pos = data.get('position_transformed')
                if pos is not None:
                    frame_players.append(pos)
        if len(frame_players) >= 10:
            # Take only the 10 outfield players (exclude GK)
            # GK should be the deepest player (min or max x depending on direction)
            sorted_by_x = sorted(frame_players, key=lambda p: p[0])
            outfield = sorted_by_x[1:]  # Remove deepest = likely GK
            detections.append(outfield)

    if not detections:
        return None, "unknown", 0.0

    # Accumulate votes for each frame
    formation_votes = Counter()
    for positions in detections:
        if len(positions) == 10:
            form, _ = detect_formation(positions, method=method)
            if form is not None:
                formation_votes[form] += 1

    if not formation_votes:
        return None, "unknown", 0.0

    best_form, best_votes = formation_votes.most_common(1)[0]
    conf = best_votes / len(detections)
    name = KNOWN_FORMATIONS.get(best_form, "-".join(str(c) for c in best_form))
    return best_form, name, conf
