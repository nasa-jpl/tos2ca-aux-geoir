import sys
from unittest.mock import MagicMock

# Mock fortracc_module (external dep) before importing production code.
for _mod in ("fortracc_module", "fortracc_module.objects", "fortracc_module.flow"):
    sys.modules[_mod] = MagicMock()

import numpy as np
import pandas as pd
import pytest

from auxgeoir_module.processing import (
    compute_temp_stats,
    create_initial_mask,
    filter_storms_in_memory,
    interpolate_nans_pandas,
    reconcile_single_step,
    renumber_clusters_sequential,
)


# ---------------------------------------------------------------------------
# create_initial_mask
# ---------------------------------------------------------------------------

class TestCreateInitialMask:
    def test_toggle_off_simple_threshold(self):
        t0 = np.array([[240.0, 250.0], [260.0, 230.0]])
        dummy = np.zeros_like(t0)
        mask = create_initial_mask(t0, dummy, temp_thresh=245, toggle="off")
        np.testing.assert_array_equal(mask, np.array([[True, False], [False, True]]))

    def test_toggle_on_primary_condition_passes(self):
        # Pixel below temp_thresh is always included regardless of future temp.
        t0 = np.array([[240.0]])
        t1 = np.array([[999.0]])
        mask = create_initial_mask(t0, t1, temp_thresh=245, temp_warmer_thresh=265, toggle="on")
        assert mask[0, 0] == True

    def test_toggle_on_secondary_condition_passes(self):
        # t0 < temp_warmer_thresh AND t1 <= t0-2 → included.
        t0 = np.array([[260.0]])
        t1 = np.array([[257.0]])  # 257 <= 260-2=258
        mask = create_initial_mask(t0, t1, temp_thresh=245, temp_warmer_thresh=265, toggle="on")
        assert mask[0, 0] == True

    def test_toggle_on_secondary_condition_fails_warm_future(self):
        # t0 < temp_warmer_thresh but t1 > t0-2 → excluded.
        t0 = np.array([[260.0]])
        t1 = np.array([[260.0]])  # 260 > 258
        mask = create_initial_mask(t0, t1, temp_thresh=245, temp_warmer_thresh=265, toggle="on")
        assert mask[0, 0] == False

    def test_toggle_on_above_warmer_thresh_excluded(self):
        # t0 >= temp_warmer_thresh → secondary condition cannot fire.
        t0 = np.array([[270.0]])
        t1 = np.array([[260.0]])
        mask = create_initial_mask(t0, t1, temp_thresh=245, temp_warmer_thresh=265, toggle="on")
        assert mask[0, 0] == False


# ---------------------------------------------------------------------------
# interpolate_nans_pandas
# ---------------------------------------------------------------------------

class TestInterpolateNansPandas:
    def test_no_nans_unchanged(self):
        matrix = np.array([[200.0, 220.0], [240.0, 260.0]])
        result = interpolate_nans_pandas(matrix, vmin=182, vmax=312.0)
        np.testing.assert_array_almost_equal(result, matrix)

    def test_interior_nan_filled_by_linear_interpolation(self):
        # [200, NaN, 300] → midpoint should be 250.
        matrix = np.array([[200.0, np.nan, 300.0]])
        result = interpolate_nans_pandas(matrix, vmin=182, vmax=312.0)
        assert not np.isnan(result).any()
        assert result[0, 1] == pytest.approx(250.0)

    def test_clips_below_vmin(self):
        matrix = np.array([[100.0, 200.0]])
        result = interpolate_nans_pandas(matrix, vmin=182, vmax=312.0)
        assert result[0, 0] == pytest.approx(182.0)
        assert result[0, 1] == pytest.approx(200.0)

    def test_clips_above_vmax(self):
        matrix = np.array([[200.0, 400.0]])
        result = interpolate_nans_pandas(matrix, vmin=182, vmax=312.0)
        assert result[0, 0] == pytest.approx(200.0)
        assert result[0, 1] == pytest.approx(312.0)

    def test_returns_numpy_array(self):
        result = interpolate_nans_pandas(np.array([[200.0, 250.0]]))
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# filter_storms_in_memory
# ---------------------------------------------------------------------------

def _storm_df(cluster_id, min_temp, size, n_timestamps):
    """Build a single-cluster DataFrame with n unique string timestamps."""
    ts = [f"20200101{i:04d}" for i in range(n_timestamps)]
    return pd.DataFrame({
        'cluster_id': cluster_id,
        'min_temp': float(min_temp),
        'size': int(size),
        'timestamp': ts,
    })


class TestFilterStormsInMemory:
    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({'cluster_id': [1], 'min_temp': [230.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            filter_storms_in_memory(df, temp_thresh=245, max_size_threshold=2500)

    def test_empty_df_returned_unchanged(self):
        result = filter_storms_in_memory(
            pd.DataFrame(), temp_thresh=245, max_size_threshold=2500
        )
        assert result.empty

    def test_valid_cold_storm_kept(self):
        df = _storm_df(cluster_id=1, min_temp=230, size=100, n_timestamps=7)
        result = filter_storms_in_memory(
            df, temp_thresh=245, max_size_threshold=2500, min_duration_steps=6
        )
        assert set(result['cluster_id']) == {1}
        assert len(result) == 7

    def test_storm_below_min_duration_removed(self):
        df = _storm_df(cluster_id=1, min_temp=230, size=100, n_timestamps=3)
        result = filter_storms_in_memory(
            df, temp_thresh=245, max_size_threshold=2500, min_duration_steps=6
        )
        assert result.empty

    def test_too_warm_and_too_large_storm_removed(self):
        # min_temp >= temp_thresh AND size > max_size_threshold → both filter conditions False.
        df = _storm_df(cluster_id=1, min_temp=250, size=3000, n_timestamps=7)
        result = filter_storms_in_memory(
            df, temp_thresh=245, max_size_threshold=2500, min_duration_steps=6
        )
        assert result.empty

    def test_multiple_storms_selective_filtering(self):
        valid = _storm_df(cluster_id=1, min_temp=230, size=100, n_timestamps=7)
        invalid = _storm_df(cluster_id=2, min_temp=250, size=3000, n_timestamps=7)
        df = pd.concat([valid, invalid], ignore_index=True)
        result = filter_storms_in_memory(
            df, temp_thresh=245, max_size_threshold=2500, min_duration_steps=6
        )
        assert set(result['cluster_id']) == {1}


# ---------------------------------------------------------------------------
# renumber_clusters_sequential
# ---------------------------------------------------------------------------

class TestRenumberClustersSequential:
    def test_empty_df_returns_unchanged_offset(self):
        result_df, new_offset = renumber_clusters_sequential(pd.DataFrame(), offset=5)
        assert result_df.empty
        assert new_offset == 5

    def test_sequential_ids_start_at_offset_plus_one(self):
        df = pd.DataFrame({
            'cluster_id': [10, 20, 30],
            'timestamp': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
        })
        result_df, new_offset = renumber_clusters_sequential(df, offset=0)
        assert sorted(result_df['cluster_id'].tolist()) == [1, 2, 3]
        assert new_offset == 3

    def test_offset_shifts_ids(self):
        df = pd.DataFrame({
            'cluster_id': [1, 2],
            'timestamp': pd.to_datetime(['2020-01-01', '2020-01-02']),
        })
        result_df, new_offset = renumber_clusters_sequential(df, offset=10)
        assert sorted(result_df['cluster_id'].tolist()) == [11, 12]
        assert new_offset == 12

    def test_sorted_by_timestamp_before_renumbering(self):
        df = pd.DataFrame({
            'cluster_id': [99, 99, 99],
            'timestamp': pd.to_datetime(['2020-01-03', '2020-01-01', '2020-01-02']),
        })
        result_df, _ = renumber_clusters_sequential(df, offset=0)
        result_df = result_df.reset_index(drop=True)
        assert result_df.loc[0, 'timestamp'] == pd.Timestamp('2020-01-01')
        assert result_df.loc[0, 'cluster_id'] == 1
        assert result_df.loc[2, 'timestamp'] == pd.Timestamp('2020-01-03')
        assert result_df.loc[2, 'cluster_id'] == 3


# ---------------------------------------------------------------------------
# compute_temp_stats
# ---------------------------------------------------------------------------

class TestComputeTempStats:
    def test_known_array_min_and_mean(self):
        temps = np.array([200.0, 220.0, 240.0, 210.0])
        min_temp, mean_temp = compute_temp_stats(temps)
        assert min_temp == pytest.approx(200.0)
        assert mean_temp == pytest.approx(217.5)

    def test_single_element(self):
        temps = np.array([230.0])
        min_temp, mean_temp = compute_temp_stats(temps)
        assert min_temp == pytest.approx(230.0)
        assert mean_temp == pytest.approx(230.0)

    def test_uniform_array(self):
        temps = np.full(5, 245.0)
        min_temp, mean_temp = compute_temp_stats(temps)
        assert min_temp == pytest.approx(245.0)
        assert mean_temp == pytest.approx(245.0)

    def test_returns_two_values(self):
        result = compute_temp_stats(np.array([200.0, 250.0, 300.0]))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# reconcile_single_step
# ---------------------------------------------------------------------------

def _cluster_row(cluster_id, xs, ys, size=None):
    """Single-row DataFrame representing one cluster at one timestep."""
    if size is None:
        size = len(xs)
    return pd.DataFrame([{
        'cluster_id': cluster_id,
        'x_coords': np.array(xs, dtype=np.int32),
        'y_coords': np.array(ys, dtype=np.int32),
        'size': size,
    }])


class TestReconcileSingleStep:
    def test_empty_current_returns_inputs_unchanged(self):
        prev = _cluster_row(1, [0, 1], [0, 0])
        result_df, result_prev = reconcile_single_step(
            pd.DataFrame(), prev, overlap_percentage=-1
        )
        assert result_df.empty
        pd.testing.assert_frame_equal(result_prev, prev)

    def test_empty_prev_returns_inputs_unchanged(self):
        current = _cluster_row(1, [0, 1], [0, 0])
        result_df, result_prev = reconcile_single_step(
            current, pd.DataFrame(), overlap_percentage=-1
        )
        pd.testing.assert_frame_equal(result_df, current)
        assert result_prev.empty

    def test_one_to_one_match_inherits_prev_id(self):
        prev = _cluster_row(42, [0, 1, 2], [0, 0, 0])
        current = _cluster_row(99, [0, 1, 2], [0, 0, 0])  # identical coords
        result_df, _ = reconcile_single_step(current, prev, overlap_percentage=-1)
        assert result_df.iloc[0]['cluster_id'] == 42

    def test_no_overlap_treated_as_new_storm(self):
        prev = _cluster_row(1, [0, 1], [0, 0])
        current = _cluster_row(2, [10, 11], [10, 10])  # completely disjoint
        result_df, _ = reconcile_single_step(current, prev, overlap_percentage=-1)
        assert result_df.iloc[0]['cluster_id'] == 2  # keeps own id

    def test_split_both_children_inherit_prev_id(self):
        # One prev cluster → two current clusters, both overlapping it.
        prev = _cluster_row(1, [0, 1, 2, 3], [0, 0, 0, 0], size=4)
        current = pd.concat([
            _cluster_row(10, [0, 1], [0, 0]),
            _cluster_row(11, [2, 3], [0, 0]),
        ], ignore_index=True)
        result_df, _ = reconcile_single_step(current, prev, overlap_percentage=-1)
        assert set(result_df['cluster_id']) == {1}
        assert len(result_df) == 2

    def test_current_overlapping_multiple_prev_inherits_largest(self):
        # With overlap_percentage < 0, current matches the largest overlapping prev.
        prev = pd.concat([
            _cluster_row(1, [0, 1], [0, 0], size=2),
            _cluster_row(2, [2, 3], [0, 0], size=4),
        ], ignore_index=True)
        current = _cluster_row(99, [0, 1, 2, 3], [0, 0, 0, 0])
        result_df, _ = reconcile_single_step(current, prev, overlap_percentage=-1)
        assert result_df.iloc[0]['cluster_id'] == 2  # largest prev wins

    def test_percentage_overlap_weak_match_treated_as_new(self):
        # Prev has 10 pixels; current shares only 1 → 10% < 50% threshold → no match.
        prev = _cluster_row(1, list(range(10)), [0] * 10, size=10)
        current = _cluster_row(2, [0, 20, 21], [0, 5, 6])
        result_df, _ = reconcile_single_step(current, prev, overlap_percentage=0.5)
        assert result_df.iloc[0]['cluster_id'] == 2

    def test_percentage_overlap_strong_match_inherits_prev_id(self):
        # Prev has 4 pixels; current shares 3 → 75% ≥ 50% threshold → match.
        prev = _cluster_row(1, [0, 1, 2, 3], [0, 0, 0, 0], size=4)
        current = _cluster_row(2, [0, 1, 2, 10], [0, 0, 0, 0])
        result_df, _ = reconcile_single_step(current, prev, overlap_percentage=0.5)
        assert result_df.iloc[0]['cluster_id'] == 1
