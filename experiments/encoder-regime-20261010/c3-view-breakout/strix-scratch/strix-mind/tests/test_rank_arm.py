import json
import random
import runpy
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM = runpy.run_path(str(ROOT / 'lib/rank_arm.py'))
NativeQuant = runpy.run_path(str(ROOT / 'lib/native_quant.py'))['NativeQuant']


class Mutating:
    def __init__(self, inner, rung, mutate):
        self.inner, self.rung, self.mutate = inner, rung, mutate
        self.library_ref = inner.library_ref

    def pack(self, values, level):
        return self.inner.pack(values, level)

    def unpack(self, packet):
        decoded = self.inner.unpack(packet)
        return self.mutate(decoded) if packet.level == self.rung else decoded


def shuffled(values):
    order = list(range(len(values)))
    random.Random(7).shuffle(order)
    return [values[i] for i in order]


class RankArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quantizer = NativeQuant(ARM['LOCK'])
        cls.result = ARM['measure'](cls.quantizer)

    def test_real_codecs_pass_all_arms_and_record_every_rung(self):
        result = self.result
        self.assertTrue(result['passed'], result['arms'])
        self.assertEqual(result['rows']['fp32']['overlap'], 1.0)
        self.assertGreaterEqual(result['rows']['8bit']['overlap'], ARM['FLOOR'])
        self.assertLess(result['rows']['8bit_shuffled']['overlap'], ARM['FLOOR'])
        for rung in ('p2', 'p1'):
            self.assertIn('overlap', result['rows'][rung])

    def test_shuffled_q8_decode_fails_the_floor_arm(self):
        result = ARM['measure'](Mutating(self.quantizer, '8bit', shuffled))
        self.assertFalse(result['arms']['q8_floor'])
        self.assertFalse(result['passed'])

    def test_perturbed_fp32_decode_fails_the_identity_arm(self):
        bumped = lambda values: [v + (0.5 if i == 0 else 0.0) for i, v in enumerate(values)]
        result = ARM['measure'](Mutating(self.quantizer, 'fp32', bumped))
        self.assertFalse(result['arms']['identity'])

    def test_receipt_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / 'arm.json'
            self.assertEqual(ARM['main'](['--receipt', str(receipt)]), 0)
            self.assertTrue(json.loads(receipt.read_text())['passed'])
            with self.assertRaises(FileExistsError):
                ARM['main'](['--receipt', str(receipt)])


class CandidateArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parents = ARM['fixture']()
        cls.views = [ARM['scoring'](p) for p in cls.parents]
        cls.result = ARM['judge'](cls.views, cls.parents)

    def test_reference_projection_passes_on_floor_margin_and_shuffle(self):
        result = self.result
        self.assertTrue(result['passed'], result['rows'])
        self.assertGreaterEqual(result['rows']['candidate'], ARM['FLOOR'])
        self.assertAlmostEqual(result['rows']['margin_over_random_projection'], 0.356, delta=0.001)
        self.assertLess(result['rows']['shuffled_across_anchors'], ARM['FLOOR'])
        self.assertIn('random_projection_encoder', result['rows'])

    def test_random_encoder_as_candidate_fails_at_margin_zero(self):
        result = ARM['judge'](ARM['random_encoder'](self.parents), self.parents)
        self.assertEqual(result['rows']['margin_over_random_projection'], 0.0)
        self.assertFalse(result['arms']['margin_over_random_projection'])
        self.assertFalse(result['passed'])

    def test_external_float32_files_match_in_memory_judgement(self):
        rows = 64
        with tempfile.TemporaryDirectory() as directory:
            views, parents = Path(directory) / 'views.f32', Path(directory) / 'parents.f32'
            views.write_bytes(b''.join(struct.pack('<128f', *v) for v in self.views[:rows]))
            parents.write_bytes(b''.join(struct.pack('<768f', *p) for p in self.parents[:rows]))
            receipt = Path(directory) / 'arm.json'
            ARM['main'](['--candidates', str(views), '--references', str(parents), '--receipt', str(receipt)])
            expected = ARM['judge'](ARM['read_f32'](views, 128), ARM['read_f32'](parents, 768))
            self.assertEqual(json.loads(receipt.read_text())['rows'], expected['rows'])
            with self.assertRaises(ValueError):
                ARM['judge'](self.parents[:rows], self.parents[:rows])

    def test_views_are_judged_within_the_byte_budget(self):
        wide = [ARM['scoring'](p, 320) for p in self.parents]
        result = ARM['judge'](wide, self.parents, element_bytes=1)
        self.assertEqual(result['rows']['view_bytes'], 320)
        self.assertTrue(result['arms']['within_budget'])
        self.assertTrue(result['passed'], result['rows'])
        over = [ARM['scoring'](p, 513) for p in self.parents]
        result = ARM['judge'](over, self.parents, element_bytes=1)
        self.assertEqual(result['rows']['view_bytes'], 513)
        self.assertFalse(result['arms']['within_budget'])
        self.assertFalse(result['passed'])
        self.assertTrue(ARM['judge'](self.views, self.parents)['arms']['within_budget'])
        self.assertFalse(ARM['judge']([ARM['scoring'](p, 129) for p in self.parents], self.parents)['arms']['within_budget'])
        with self.assertRaises(ValueError):
            ARM['judge'](self.views, self.parents, element_bytes=3)

    def test_int8_quantization_is_applied_before_judging(self):
        quantized = ARM['quantize_int8'](self.views[:4])
        self.assertTrue(all(abs(round(v * 127) - v * 127) < 1e-9 for row in quantized for v in row))
        self.assertNotEqual(quantized, self.views[:4])

    @staticmethod
    def clustered(center, tail):
        rng = random.Random(3)
        centers = [[rng.gauss(0.0, center) for _ in range(128)] for _ in range(23)]
        return [ARM['f32']([c + rng.gauss(0.0, 0.5) for c in centers[i % 23]] + [rng.gauss(0.0, tail) for _ in range(640)])
                for i in range(256)]

    def test_strong_clusters_pass_only_when_candidate_beats_projection_by_margin(self):
        for center, tail, expected in ((1.0, 0.5, False), (0.8, 0.45, True)):
            parents = self.clustered(center, tail)
            result = ARM['judge']([ARM['scoring'](p) for p in parents], parents)
            rows = result['rows']
            self.assertGreaterEqual(rows['candidate'], ARM['FLOOR'])
            self.assertEqual(result['arms']['margin_over_random_projection'], expected, rows)
            self.assertEqual(result['passed'], expected, rows)
        self.assertGreaterEqual(rows['random_projection_encoder'], 0.8)

    def test_misaligned_sets_are_refused(self):
        with self.assertRaises(ValueError):
            ARM['agreement'](self.views[:20], self.parents[:21])


if __name__ == '__main__':
    unittest.main()
