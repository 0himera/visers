import errno
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import visers as nano


ROOT = Path(__file__).resolve().parents[3]
JPEG = ROOT / "tests" / "fixtures" / "rgb-pattern.jpg"
WEBP = ROOT / "tests" / "fixtures" / "oriented-animation.webp"
PNG = ROOT / "assets" / "domain-comparison.png"


class TestImageAPI(unittest.TestCase):
    def test_compatibility_import(self):
        import visers_nano

        for name in nano.__all__:
            self.assertIs(getattr(visers_nano, name), getattr(nano, name))

    def test_shapes_and_mixed_batch_order(self):
        self.assertEqual(nano.dimensions(), nano.DIMENSIONS)
        uploads = [JPEG.read_bytes(), PNG.read_bytes(), WEBP.read_bytes()]
        for mode in ["fast", "full"]:
            expected = np.stack([nano.encode_single(b, mode=mode) for b in uploads])
            for jobs in [1, 3]:
                images = (im for im in [str(JPEG), uploads[1], WEBP])
                rows = nano.encode(images, mode=mode, jobs=jobs)
                np.testing.assert_array_equal(rows, expected)
                self.assertEqual(rows.shape, (3, nano.DIMENSIONS))
                self.assertEqual(rows.dtype, np.float32)
                self.assertTrue(rows.flags.c_contiguous)
                np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(nano.encode(JPEG)[0], nano.encode_single(JPEG))
        self.assertEqual(nano.encode(uploads[0]).shape, (1, nano.DIMENSIONS))
        self.assertEqual(nano.encode([]).shape, (0, nano.DIMENSIONS))
        self.assertEqual(nano.encode([]).dtype, np.float32)

    def test_paths_mutable_buffers_and_owned_result(self):
        data = JPEG.read_bytes()
        expected = nano.encode_single(data)
        interleaved = bytearray(len(data) * 2)
        interleaved[::2] = data
        for image in [bytearray(data), memoryview(data), memoryview(interleaved)[::2]]:
            actual = nano.encode_single(image)
            np.testing.assert_array_equal(actual, expected)
            actual.fill(0)
        with tempfile.TemporaryDirectory() as directory:
            disguised = Path(directory) / "изображение.data"
            disguised.write_bytes(data)
            np.testing.assert_array_equal(nano.encode_single(disguised), expected)

    def test_threaded_calls_and_mode_switch(self):
        data = PNG.read_bytes()
        refs = {mode: nano.encode_single(data, mode=mode) for mode in ["fast", "full"]}
        self.assertFalse(np.array_equal(refs["fast"], refs["full"]))
        modes = ["full", "fast"] * 3
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(nano.encode, [data, JPEG], mode=mode, jobs=2) for mode in modes]
            for mode, future in zip(modes, futures):
                np.testing.assert_array_equal(future.result()[0], refs[mode])

    def test_errors_have_input_index_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jpg"
            with self.assertRaises(FileNotFoundError) as caught:
                nano.encode([JPEG, missing, b"bad"], jobs=3)
            self.assertEqual(caught.exception.errno, errno.ENOENT)
            self.assertEqual(Path(caught.exception.filename), missing)
            self.assertIn("image[1]", str(caught.exception))
            with self.assertRaisesRegex(ValueError, r"image\[0\]"):
                nano.encode([b"broken", missing], jobs=2)
        with self.assertRaisesRegex(TypeError, r"image\[1\]"):
            nano.encode([JPEG, 123])
        with self.assertRaises(TypeError):
            nano.encode_single([JPEG])
        for jobs in [0, -1]:
            with self.assertRaisesRegex(ValueError, "jobs"):
                nano.encode([], jobs=jobs)
        with self.assertRaisesRegex(ValueError, "mode"):
            nano.encode_single(JPEG, mode="other")
        self.assertEqual(nano.encode_single(JPEG).shape, (nano.DIMENSIONS,))

    def test_vector_metrics(self):
        self.assertAlmostEqual(nano.similarity([3, 4], [4, 3]), 0.96)
        self.assertEqual(nano.distance([0, 0], [3, 4]), 5.0)
        view = np.arange(9, dtype=np.float32)[::-2]
        self.assertAlmostEqual(nano.similarity(view, view.copy()), 1.0)
        self.assertEqual(nano.distance(view, view.astype(np.float64)), 0.0)
        for fn in [nano.similarity, nano.distance]:
            for a, b in [([], []), ([1], [1, 2]), ([[1, 2]], [1, 2]), ([np.nan], [0]), ([0], [np.inf])]:
                with self.assertRaises(ValueError):
                    fn(a, b)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            nano.similarity([0], [1])

    @unittest.skipUnless(os.environ.get("VISERS_NANO_CLI"), "set VISERS_NANO_CLI to the matching Rust binary")
    def test_bitwise_parity_with_rust_cli(self):
        paths = [JPEG, PNG, WEBP]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vectors.f32"
            for mode in ["fast", "full"]:
                flags = ["--full"] if mode == "full" else []
                subprocess.run(
                    [os.environ["VISERS_NANO_CLI"], "--jobs", "2", *flags, "-o", output, *paths],
                    check=True,
                )
                cli = np.fromfile(output, dtype="<f4").reshape(len(paths), nano.DIMENSIONS)
                rows = nano.encode(paths, mode=mode, jobs=2)
                np.testing.assert_array_equal(rows.view(np.uint32), cli.view(np.uint32))


if __name__ == "__main__":
    unittest.main()
