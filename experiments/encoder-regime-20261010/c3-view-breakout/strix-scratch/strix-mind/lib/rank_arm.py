"""Vector-reading acceptance arm: cosine top-k rank of original vs decoded views within a byte budget (ruling 60: 512 bytes per view)."""
from __future__ import annotations

import argparse
import json
import math
import random
import runpy
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT.parent / 'maxwell/docs/data-context/agi-model/heads/agi0-served-lock.json'

SEED = 20261010
ANCHORS = 256
QUERIES = 32
NOISE = 0.35
K = 10
FLOOR = 0.9
MARGIN = 0.1
RUNGS = ('fp32', '8bit', 'p2', 'p1')


def f32(values):
    return list(struct.unpack('<%df' % len(values), struct.pack('<%df' % len(values), *values)))


def corpus(seed=SEED):
    rng = random.Random(seed)
    anchors = [f32([rng.gauss(0.0, 1.0) for _ in range(128)]) for _ in range(ANCHORS)]
    picks = rng.sample(range(ANCHORS), QUERIES)
    queries = [f32([a + rng.gauss(0.0, NOISE) for a in anchors[i]]) for i in picks]
    permutation = list(range(128))
    rng.shuffle(permutation)
    return anchors, queries, permutation


def top_k(query, anchors, norms, k=K):
    qn = math.sqrt(sum(v * v for v in query))
    scores = [sum(a * b for a, b in zip(query, anchor)) / (qn * n) for anchor, n in zip(anchors, norms)]
    return [i for _, i in sorted((-s, i) for i, s in enumerate(scores))[:k]]


def overlap(reference, candidate, anchors, norms):
    total, ordered = 0.0, True
    for original, decoded in zip(reference, candidate):
        a, b = top_k(original, anchors, norms), top_k(decoded, anchors, norms)
        total += len(set(a) & set(b)) / K
        ordered = ordered and a == b
    return total / len(reference), ordered


def measure(quantizer, seed=SEED):
    anchors, queries, permutation = corpus(seed)
    norms = [math.sqrt(sum(v * v for v in a)) for a in anchors]
    decoded = {rung: [quantizer.unpack(quantizer.pack(q, rung)) for q in queries] for rung in RUNGS}
    rows = {}
    for rung in RUNGS:
        value, ordered = overlap(queries, decoded[rung], anchors, norms)
        rows[rung] = {'overlap': value, 'order_equal': ordered, 'passes_floor': value >= FLOOR}
    shuffled = [[d[p] for p in permutation] for d in decoded['8bit']]
    value, ordered = overlap(queries, shuffled, anchors, norms)
    rows['8bit_shuffled'] = {'overlap': value, 'order_equal': ordered, 'passes_floor': value >= FLOOR}
    arms = {'identity': rows['fp32']['overlap'] == 1.0 and rows['fp32']['order_equal'],
            'q8_floor': rows['8bit']['passes_floor'],
            'shuffled_control_below_floor': not rows['8bit_shuffled']['passes_floor']}
    return {'schema': 'strix.rank-arm/v1', 'seed': seed, 'anchors': ANCHORS, 'queries': QUERIES,
            'noise': NOISE, 'k': K, 'floor': FLOOR, 'chance_overlap': K / ANCHORS,
            'library_ref': quantizer.library_ref, 'anchor_set': 'seeded-synthetic',
            'rows': rows, 'arms': arms, 'passed': all(arms.values())}


VIEW = 128
PARENT = 768
BUDGET_BYTES = 512
ELEMENT_BYTES = (1, 2, 4)
CLUSTERS = 23
CENTER = 0.6
SPREAD = 0.5
TAIL = 0.4


def scoring(parent, dimension=VIEW):
    """Mirror of CARL Carrier.scoring: normalized prefix."""
    norm = math.hypot(*parent[:dimension])
    return [v / norm for v in parent[:dimension]]


def neighbours(rows, k=K):
    norms = [math.sqrt(sum(v * v for v in r)) for r in rows]
    out = []
    for i, query in enumerate(rows):
        scores = ((-sum(a * b for a, b in zip(query, other)) / (norms[i] * norms[j]), j)
                  for j, other in enumerate(rows) if j != i)
        out.append({j for _, j in sorted(scores)[:k]})
    return out


def agreement(candidates, references, k=K):
    if len(candidates) != len(references) or len(candidates) <= k:
        raise ValueError('candidate and reference sets must align and exceed k')
    width = len(candidates[0])
    if width < 1 or width >= PARENT or any(len(c) != width for c in candidates) or any(len(r) != PARENT for r in references):
        raise ValueError('candidate views must share one width below 768 and parents must be 768 wide')
    if any(not math.isfinite(v) for row in candidates + references for v in row):
        raise ValueError('vectors must be finite')
    a, b = neighbours(candidates, k), neighbours(references, k)
    return sum(len(x & y) for x, y in zip(a, b)) / (k * len(a))


def quantize_int8(rows):
    """Symmetric per-row int8: the bytes a 1-byte view actually stores."""
    out = []
    for row in rows:
        scale = max(abs(v) for v in row) or 1.0
        out.append([round(v / scale * 127) / 127 for v in row])
    return out


def random_encoder(references, seed=SEED, width=VIEW):
    rng = random.Random(seed ^ 0x5EED)
    matrix = [[rng.gauss(0.0, 1.0) for _ in range(PARENT)] for _ in range(width)]
    return [[sum(w * v for w, v in zip(row, parent)) for row in matrix] for parent in references]


def judge(candidates, references, seed=SEED, element_bytes=4, budget_bytes=BUDGET_BYTES):
    if element_bytes not in ELEMENT_BYTES:
        raise ValueError('element_bytes must be 1, 2 or 4')
    width = len(candidates[0]) if candidates else 0
    if element_bytes == 1:
        candidates = quantize_int8(candidates)
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)
    rows = {'candidate': agreement(candidates, references),
            'shuffled_across_anchors': agreement([candidates[i] for i in order], references),
            'random_projection_encoder': agreement(random_encoder(references, seed, width), references),
            'view_width': width, 'element_bytes': element_bytes, 'view_bytes': width * element_bytes, 'budget_bytes': budget_bytes}
    rows['margin_over_random_projection'] = rows['candidate'] - rows['random_projection_encoder']
    arms = {'candidate_floor': rows['candidate'] >= FLOOR,
            'margin_over_random_projection': rows['margin_over_random_projection'] >= MARGIN,
            'shuffled_below_floor': rows['shuffled_across_anchors'] < FLOOR,
            'within_budget': rows['view_bytes'] <= budget_bytes}
    return {'schema': 'strix.rank-arm-candidate/v3', 'seed': seed, 'anchors': len(candidates), 'k': K,
            'floor': FLOOR, 'margin': MARGIN, 'chance_overlap': K / (len(candidates) - 1),
            'rows': rows, 'arms': arms, 'passed': all(arms.values())}


def fixture(seed=SEED, count=ANCHORS):
    """Parents whose structure sits in the 128 prefix; the 640 tail is independent noise."""
    rng = random.Random(seed)
    centers = [[rng.gauss(0.0, CENTER) for _ in range(VIEW)] for _ in range(CLUSTERS)]
    parents = []
    for i in range(count):
        center = centers[i % CLUSTERS]
        head = [c + rng.gauss(0.0, SPREAD) for c in center]
        parents.append(f32(head + [rng.gauss(0.0, TAIL) for _ in range(PARENT - VIEW)]))
    return parents


def read_f32(path, width):
    raw = path.read_bytes()
    if not raw or len(raw) % (4 * width):
        raise ValueError('%s is not a whole number of %d-wide float32 rows' % (path, width))
    values = struct.unpack('<%df' % (len(raw) // 4), raw)
    return [list(values[i:i + width]) for i in range(0, len(values), width)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=LOCK)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--candidates', type=Path, help='n x width float32 little-endian rows')
    parser.add_argument('--width', type=int, default=VIEW, help='candidate view width')
    parser.add_argument('--element-bytes', type=int, default=4, choices=ELEMENT_BYTES, help='bytes per stored element; 1 quantizes to int8 before judging')
    parser.add_argument('--budget-bytes', type=int, default=BUDGET_BYTES)
    parser.add_argument('--references', type=Path, help='n x 768 float32 little-endian parent rows')
    parser.add_argument('--fixture', action='store_true', help='judge the scoring(128) projection of the seeded fixture')
    args = parser.parse_args(argv)
    if (args.candidates is None) != (args.references is None):
        parser.error('--candidates and --references go together')
    if args.candidates is not None:
        result = judge(read_f32(args.candidates, args.width), read_f32(args.references, PARENT), element_bytes=args.element_bytes, budget_bytes=args.budget_bytes)
    elif args.fixture:
        parents = fixture()
        result = judge([scoring(p) for p in parents], parents)
        result['anchor_set'] = 'seeded-synthetic-prefix-structured'
        result['fixture'] = {'clusters': CLUSTERS, 'center': CENTER, 'spread': SPREAD, 'tail': TAIL}
    else:
        quantizer = runpy.run_path(str(ROOT / 'lib/native_quant.py'))['NativeQuant'](args.lock)
        result = measure(quantizer)
    body = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.receipt:
        with args.receipt.open('x') as out:
            out.write(body)
    sys.stdout.write(body)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
