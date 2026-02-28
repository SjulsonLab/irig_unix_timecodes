import numpy as np
import pytest

from neurokairos.decoders.ttl import auto_threshold, detect_edges, measure_pulse_widths


class TestAutoThreshold:
    def test_clean_bimodal(self):
        low = np.zeros(5000)
        high = np.full(5000, 10000.0)
        signal = np.concatenate([low, high])
        thresh = auto_threshold(signal)
        assert 1000 < thresh < 9000

    def test_noisy_bimodal(self):
        rng = np.random.default_rng(0)
        low = rng.normal(0, 500, 5000)
        high = rng.normal(10000, 500, 5000)
        signal = np.concatenate([low, high])
        thresh = auto_threshold(signal)
        # Threshold should land between the two peaks
        assert 2000 < thresh < 8000

    def test_unequal_populations(self):
        # 90% low, 10% high — threshold should still separate them
        rng = np.random.default_rng(1)
        low = rng.normal(0, 200, 9000)
        high = rng.normal(10000, 200, 1000)
        signal = np.concatenate([low, high])
        thresh = auto_threshold(signal)
        assert 1000 < thresh < 9000


class TestDetectEdges:
    def test_single_pulse(self):
        signal = np.zeros(100, dtype=np.float64)
        signal[30:60] = 10000.0
        rising, falling = detect_edges(signal, 5000.0)
        np.testing.assert_array_equal(rising, [30])
        np.testing.assert_array_equal(falling, [60])

    def test_multiple_pulses(self):
        signal = np.zeros(300, dtype=np.float64)
        signal[10:50] = 10000.0
        signal[100:200] = 10000.0
        signal[250:280] = 10000.0
        rising, falling = detect_edges(signal, 5000.0)
        np.testing.assert_array_equal(rising, [10, 100, 250])
        np.testing.assert_array_equal(falling, [50, 200, 280])

    def test_starts_high(self):
        signal = np.zeros(100, dtype=np.float64)
        signal[0:40] = 10000.0
        rising, falling = detect_edges(signal, 5000.0)
        # Signal starts HIGH — this is a partial pulse (recording started
        # mid-pulse). There is no real rising edge, only a falling edge.
        # The orphaned falling edge is discarded by measure_pulse_widths.
        np.testing.assert_array_equal(rising, [])
        np.testing.assert_array_equal(falling, [40])

    def test_ends_high(self):
        signal = np.zeros(100, dtype=np.float64)
        signal[60:100] = 10000.0
        rising, falling = detect_edges(signal, 5000.0)
        np.testing.assert_array_equal(rising, [60])
        np.testing.assert_array_equal(falling, [])


class TestMeasurePulseWidths:
    def test_basic(self):
        rising = np.array([10, 100, 250])
        falling = np.array([50, 200, 280])
        onsets, widths = measure_pulse_widths(rising, falling)
        np.testing.assert_array_equal(onsets, [10, 100, 250])
        np.testing.assert_array_equal(widths, [40, 100, 30])

    def test_falling_before_first_rising(self):
        # A falling edge at 5 has no matching rising edge — should be discarded
        rising = np.array([30, 100])
        falling = np.array([5, 60, 150])
        onsets, widths = measure_pulse_widths(rising, falling)
        np.testing.assert_array_equal(onsets, [30, 100])
        np.testing.assert_array_equal(widths, [30, 50])

    def test_unmatched_trailing_rising(self):
        # Last rising edge has no subsequent falling edge — discarded
        rising = np.array([10, 100, 250])
        falling = np.array([50, 200])
        onsets, widths = measure_pulse_widths(rising, falling)
        np.testing.assert_array_equal(onsets, [10, 100])
        np.testing.assert_array_equal(widths, [40, 100])

    def test_empty_inputs(self):
        onsets, widths = measure_pulse_widths(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        )
        assert len(onsets) == 0
        assert len(widths) == 0
