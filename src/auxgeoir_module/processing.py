import pandas as pd
import numpy as np
import time
import warnings
import concurrent.futures
import numba
import gc

from datetime import datetime
from tqdm import tqdm
from skimage import measure
from collections import defaultdict
from fortracc_module.objects import GeoGrid, SparseGeoGrid, SparseMask
from fortracc_module.flow import SparseTimeOrderedSequence

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in reduce")
warnings.filterwarnings("ignore", category=numba.errors.NumbaDeprecationWarning)

#############################
# UTILITY FUNCTIONS
#############################

def interpolate_nans_pandas(matrix, vmin=182, vmax=312.0):
    """Interpolate NaN values in a matrix using pandas' linear interpolation."""
    df = pd.DataFrame(matrix)
    return df.interpolate(method='linear', axis=1, limit_direction='both')\
             .interpolate(method='linear', axis=0, limit_direction='both')\
             .clip(vmin, vmax).to_numpy()


def filter_storms_in_memory(
    df: pd.DataFrame,
    temp_thresh: float,
    max_size_threshold: int,
    min_duration_steps: int = 6
) -> pd.DataFrame:
    """
    Apply global storm filters to a DataFrame in memory, mimicking the
    logic of apply_global_storm_filters_memory_safe.
    """
    print("\n" + "="*60)
    print("APPLYING GLOBAL STORM FILTERS (in-memory)")
    print("="*60)

    if df.empty:
        print("  Empty DataFrame. Skipping filtering.")
        return df

    required_columns = ['cluster_id', 'min_temp', 'size', 'timestamp']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for filtering: {missing_cols}")

    # --- Pass 1: Compute storm-level stats ---
    print("  Computing storm stats for filtering...")
    storm_stats = df.groupby('cluster_id').agg(
        min_temp=('min_temp', 'min'),
        min_size=('size', 'min'),
        duration=('timestamp', 'nunique')
    )

    valid_storms = storm_stats[
        (storm_stats['min_temp'] < temp_thresh) |
        (storm_stats['min_size'] <= max_size_threshold)
    ]
    valid_storms = valid_storms[valid_storms['duration'] >= min_duration_steps]

    valid_ids = set(valid_storms.index)
    print(f"  → {len(valid_ids)} valid storms (out of {len(storm_stats)}) passed filtering.")

    # --- Pass 2: Filter the original DataFrame ---
    filtered_df = df[df['cluster_id'].isin(valid_ids)].copy()
    print(f"  → Final row count after filtering: {len(filtered_df)}")

    return filtered_df


def create_initial_mask(t0_data, t0plushalfhour_data, temp_thresh=245, temp_warmer_thresh=265, toggle="on"):
    """
    Find pixels satisfying clustering criteria.
    
    Parameters:
    -----------
    t0_data : ndarray
        Temperature data at time t
    t0plushalfhour_data : ndarray
        Temperature data at time t+30 minutes
    temp_thresh : float, default=245
        Primary temperature threshold in Kelvin
    temp_warmer_thresh : float, default=265
        Secondary warmer temperature threshold
    toggle : str, default="on"
        "on" for two-stage clustering (default behavior)
        "off" for simple one-stage clustering using only temp_thresh
        
    Returns:
    --------
    ndarray: Boolean mask of pixels meeting criteria
    """
    if toggle == "off":
        return (t0_data < temp_thresh)
    else:  # toggle is "on"
        """For example, find pixels satisfying T<temp_thresh (in K) or (T<265K & T'≤T-2K)."""
        return (t0_data < temp_thresh) | ((t0_data < temp_warmer_thresh) & (t0plushalfhour_data <= t0_data - 2))


def convert_df_to_sparse_sequence(
    df: pd.DataFrame,
    timestamps: tuple,
    grid: GeoGrid,
    cls,  # Class to instantiate, e.g., SparseTimeOrderedSequence
    default_mask_type: str = "cloud"
):
    """
    Convert a flat DataFrame (e.g., from Parquet preparation) into a SparseTimeOrderedSequence-like object.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with one row per detected object per time step.
    timestamps: tuple
        List of all timestamps for the chunk
    grid : GeoGrid
        The GeoGrid the image data is based on.
    cls : class
        Class to instantiate at the end (e.g., SparseTimeOrderedSequence)
    default_mask_type : str
        Optional mask type if none is provided.

    Returns
    -------
    An instance of `cls`, usually SparseTimeOrderedSequence
    """

    required_columns = {'cluster_id', 'timestamp', 'x_coords', 'y_coords', 'T_IR'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    grouped = df.groupby("cluster_id")
    sparse_events = []

    for cluster_id, group in grouped:
        time_series = []

        for _, row in group.iterrows():
            timestamp = row['timestamp'].strftime('%Y%m%d%H%M')
            x_coords = row['x_coords']
            y_coords = row['y_coords']
            values = row['T_IR']
            mask_type = default_mask_type

            # Optional properties
            extra_properties = {
                'size': row.get('size', None),
                'min_temp': row.get('min_temp', None),
                'mean_temp': row.get('mean_temp', None),
                'merged': row.get('merged', None),
                'merged_into': row.get('merged_into', None),
                'cluster_id': cluster_id,
            }

            # Build SparseMask with optional properties
            sparse_mask = SparseMask.from_coords(
                x_coords=x_coords,
                y_coords=y_coords,
                values=values,
                timestamp=timestamp,
                mask_type=mask_type,
                properties=extra_properties  # Assuming from_coords accepts metadata
            )

            time_series.append(sparse_mask)

        # Sort by time if needed
        time_series.sort(key=lambda m: m.timestamp)
        sparse_events.append(time_series)

    # Convert grid to sparse version
    sparse_grid = SparseGeoGrid.from_grid(grid)

    # Unique timestamps
    # ts = sorted(df['timestamp'].unique().tolist())
    # timestamps = tuple( datetime.fromtimestamp(i / 1e9).strftime('%Y%m%d%H%M') for i in ts )

    # Return final object
    return cls(
        sparse_events,
        timestamps,
        sparse_grid,
        fortracc_runner=None,
        detector_type="from_dataframe"
    )

#############################
# INITIAL CLUSTERING FUNCTIONS
#############################

@numba.jit(nopython=True)
def compute_temp_stats(temps):
    """Compute minimum and mean temperature using Numba acceleration."""
    return np.min(temps), np.mean(temps)


def process_clusters_with_stages_optimized(t0_data, initial_mask, max_size_threshold=2500, min_size=81, temp_thresh=245, toggle="on"):
    """
    
    Parameters:
    -----------
    t0_data : ndarray
        Temperature data at time t
    initial_mask : ndarray
        Boolean mask from original conditions
    max_size_threshold : int
        Size above which clusters are split (only used if toggle is "on")
    min_size : int
        Minimum cluster size to retain
    temp_thresh : float
        Temperature threshold in Kelvin
    toggle : str
        "on" for two-stage clustering, "off" for single-stage clustering
        
    Returns:
    --------
    tuple: (initial_df, refined_df) containing cluster information
    """
    # Stage 1: Initial clustering using scikit-image's connected components
    initial_clusters = measure.label(initial_mask, connectivity=2)
    props = measure.regionprops(initial_clusters)
    
    # Filter regions by size and extract properties efficiently
    initial_info = []
    small_clusters_info = []
    large_clusters_info = []
    
    for i, prop in enumerate(props):
        if prop.area >= min_size:
            cluster_id = len(initial_info) + 1
            
            # Get coordinates
            coords = prop.coords
            y_coords = coords[:, 0].astype(np.int32)
            x_coords = coords[:, 1].astype(np.int32)

            # Get temperature values for this cluster
            cluster_temps = t0_data[y_coords, x_coords]
            min_temp, mean_temp = compute_temp_stats(cluster_temps)
            
            cluster_info = {
                'cluster_id': cluster_id,
                'x_coords': x_coords,
                'y_coords': y_coords,
                'size': prop.area,
                'min_temp': min_temp,
                'mean_temp': mean_temp,
                'bbox': prop.bbox,
                'T_IR': cluster_temps 
            }
            
            initial_info.append(cluster_info)
            
            # Separate small and large clusters for more efficient processing
            # Only separate if toggle is "on", otherwise treat all clusters as small
            if toggle == "on" and prop.area > max_size_threshold:
                large_clusters_info.append(cluster_info)
            else:
                small_clusters_info.append(cluster_info)
    
    # Add small clusters directly to refined_info
    refined_info = []
    for cluster in small_clusters_info:
        cluster_dict = {k: v for k, v in cluster.items() if k != 'bbox'}
        cluster_dict['is_subcluster'] = False
        cluster_dict['parent_id'] = None
        refined_info.append(cluster_dict)
    
    # Set the next ID for subclusters
    max_initial_id = len(initial_info)
    next_id = max_initial_id + 1
    
    # Process large clusters in parallel only if toggle is "on"
    if toggle == "on" and large_clusters_info:
        # Define a worker function for parallel processing
        def process_large_cluster(cluster):
            results = []
            
            y_min, x_min, y_max, x_max = cluster['bbox']
            local_shape = (y_max - y_min, x_max - x_min)
            local_mask = np.zeros(local_shape, dtype=bool)
            
            local_y = cluster['y_coords'] - y_min
            local_x = cluster['x_coords'] - x_min
            local_mask[local_y, local_x] = True
            
            local_t0_data = t0_data[y_min:y_max, x_min:x_max]
            strict_mask = local_mask & (local_t0_data < temp_thresh)
            
            if np.any(strict_mask):
                subclusters = measure.label(strict_mask, connectivity=2)
                sub_props = measure.regionprops(subclusters)
                valid_subclusters = [p for p in sub_props if p.area >= min_size]
                
                if valid_subclusters:
                    for prop in valid_subclusters:
                        sub_coords = prop.coords
                        global_y = (sub_coords[:, 0] + y_min).astype(np.int32)
                        global_x = (sub_coords[:, 1] + x_min).astype(np.int32)
                        sub_temps = t0_data[global_y, global_x]
                        min_temp, mean_temp = compute_temp_stats(sub_temps)
                        
                        results.append({
                            'cluster_id': None,
                            'x_coords': global_x,
                            'y_coords': global_y,
                            'size': prop.area,
                            'parent_id': cluster['cluster_id'],
                            'min_temp': min_temp,
                            'mean_temp': mean_temp,
                            'is_subcluster': True,
                            'T_IR': sub_temps
                        })
                else:
                    cluster_dict = {k: v for k, v in cluster.items() if k != 'bbox'}
                    cluster_dict['is_subcluster'] = False
                    cluster_dict['parent_id'] = None
                    results.append(cluster_dict)
            else:
                cluster_dict = {k: v for k, v in cluster.items() if k != 'bbox'}
                cluster_dict['is_subcluster'] = False
                cluster_dict['parent_id'] = None
                results.append(cluster_dict)
            
            return results
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_results = list(executor.map(process_large_cluster, large_clusters_info))
        
        # Flatten results and assign IDs
        for result_list in all_results:
            for result in result_list:
                if result['is_subcluster']:
                    result['cluster_id'] = next_id
                    next_id += 1
                refined_info.append(result)
    
    initial_df = pd.DataFrame([{k: v for k, v in c.items() if k != 'bbox'} for c in initial_info])
    refined_df = pd.DataFrame(refined_info)
    
    return initial_df, refined_df

#############################
# PARALLEL PROCESSING FUNCTIONS FOR LINUX HPC
#############################

def worker_process_in_memory_for_tos2ca(args):
    """
    Worker process that performs initial clustering for a single timestep
    and returns the resulting DataFrame in memory.
    """
    image1, ts_idx1, timestamp1, image2, ts_idx2, timestamp2, temp_thresh, temp_warmer_thresh, min_size, max_size_threshold, toggle = args
    
    try:
        t0_brightness_temp = image1
        t0_interpolated = interpolate_nans_pandas(t0_brightness_temp)
        # t0_data = t0_interpolated[::-1]
        t0_data = t0_interpolated
        if toggle == "on":
            t0_plus_halfhour_brightness_temp = image2
            t0_plus_halfhour_interpolated = interpolate_nans_pandas(t0_plus_halfhour_brightness_temp)
            # t0plushalfhour_data = t0_plus_halfhour_interpolated[::-1]
            t0plushalfhour_data = t0_plus_halfhour_interpolated
        else:
            t0plushalfhour_data = t0_data


        initial_mask = create_initial_mask(t0_data, t0plushalfhour_data, temp_thresh, temp_warmer_thresh, toggle)
        
        _, refined_df = process_clusters_with_stages_optimized(
            t0_data, initial_mask, 
            min_size=min_size,
            max_size_threshold=max_size_threshold,
            temp_thresh=temp_thresh,
            toggle=toggle
        )
        
        timestamp = datetime.strptime(timestamp1, '%Y%m%d%H%M')
        
        if not refined_df.empty:
            refined_df['timestamp'] = timestamp
        
        return refined_df

    except Exception as e:
        # Log the error but return an empty DataFrame to not halt the whole process
        print(f"Error in worker for {timestamp1} (t={ts_idx1}): {e}")
        return pd.DataFrame()


#############################
# DATELINE RECONCILIATION FUNCTIONS
#############################

def reconcile_dateline_clusters(refined_df, lon_array_f32, lat_array_f32):
    """
    In‐place dateline reconciliation, using a pandas.Series for edge_mask so that
    dropping rows stays in sync.

    Parameters
    ----------
    refined_df : pd.DataFrame
        Must contain columns ['x_coords', 'y_coords', 'T_IR', 'timestamp', 'size'], etc.
        Each row’s 'x_coords' is a 1D np.ndarray of integer indices into lon_array_f32.
    lon_array_f32 : 1D numpy array of dtype float32
    lat_array_f32 : 1D numpy array of dtype float32

    Returns
    -------
    pd.DataFrame
        The same DataFrame object passed in (modified in place), with dateline‐merged clusters removed.
    """

    start_time = time.time()
    print("Starting dateline reconciliation…")

    # 1) Identify “west_edge” and “east_edge”
    west_edge = float(lon_array_f32.min())
    east_edge = float(lon_array_f32.max())

    # 2) Build a pandas.Series mask of length == len(refined_df), initially False
    edge_mask = pd.Series(False, index=refined_df.index)

    # 3) Instead of a Python loop, compute per‐row min and max longitudes in one pass:
    #    – For each row of refined_df, pull out the x_coords array, index into lon_array_f32, and take min() and max().
    #    – Then compare those extremes to west_edge/east_edge using np.isclose(…, atol=1e-10).
    def compute_extremes(xcoords: np.ndarray):
        if xcoords.size == 0:
            return np.inf, -np.inf
        lons = lon_array_f32[xcoords]
        return lons.min(), lons.max()

    # Use map to avoid building a giant intermediate list of arrays.  This returns a generator of (min_lon, max_lon)
    extremes = refined_df['x_coords'].map(compute_extremes)
    # Now unpack into two NumPy arrays: min_lons, max_lons
    min_lons = np.fromiter((m for m, M in extremes), dtype=np.float32, count=len(refined_df))
    max_lons = np.fromiter((M for m, M in extremes), dtype=np.float32, count=len(refined_df))

    # 4) Build the boolean Series “edge_mask” all at once
    close_to_west = np.isclose(min_lons, west_edge, atol=1e-10)
    close_to_east = np.isclose(max_lons, east_edge, atol=1e-10)
    combined_edge = close_to_west | close_to_east

    # Assign into our pandas.Series:
    edge_mask[:] = combined_edge
    print(f"  Found {edge_mask.sum()} edge clusters out of {len(refined_df)} total rows.")

    # 5) Loop over each timestamp, merging/dropping in place
    unique_timestamps = sorted(refined_df['timestamp'].unique())
    for timestamp in unique_timestamps:
        # Build the “edge rows at this timestamp” mask
        ts_mask = (refined_df['timestamp'] == timestamp) & edge_mask
        ts_indices = refined_df.index[ts_mask].tolist()
        if not ts_indices:
            continue
        # Build previous‐timestamp bboxes for only those indices that survived to this timestamp.
        prev_bboxes = {}
        for idx in ts_indices:
            row = refined_df.loc[idx]
            xs = row['x_coords']
            ys = row['y_coords']
            # Compute bounding box in x,y‐space
            min_x, max_x = float(xs.min()), float(xs.max())
            min_y, max_y = float(ys.min()), float(ys.max())
            prev_bboxes[idx] = (min_x, min_y, max_x, max_y)
        # If you want a cell‐size heuristic that depends on cluster size:
        widths  = [bbox[2] - bbox[0] for bbox in prev_bboxes.values()]
        heights = [bbox[3] - bbox[1] for bbox in prev_bboxes.values()]
        if widths and heights:
            avg_dim = max(np.mean(widths), np.mean(heights)) * 2
            cell_size = max(10, min(50, int(avg_dim)))
        else:
            cell_size = 30
        safety_margin = 2

        # Build a spatial grid mapping from (cell_x,cell_y) to list of cluster‐IDs at that timestamp
        spatial_grid = defaultdict(list)
        for idx, bbox in prev_bboxes.items():
            min_x, min_y, max_x, max_y = bbox
            min_cell_x = int((min_x - safety_margin) // cell_size)
            min_cell_y = int((min_y - safety_margin) // cell_size)
            max_cell_x = int((max_x + safety_margin) // cell_size)
            max_cell_y = int((max_y + safety_margin) // cell_size)
            for cx in range(min_cell_x, max_cell_x + 1):
                for cy in range(min_cell_y, max_cell_y + 1):
                    spatial_grid[(cx, cy)].append(idx)

        # Now scan each grid‐cell’s list for overlapping bounding boxes (i.e., clusters to merge)
        merged_indices = set()
        for idx in ts_indices:
            if idx in merged_indices:
                continue
            # “c1” is the “surviving” cluster
            bbox1 = prev_bboxes[idx]
            min_x1, min_y1, max_x1, max_y1 = bbox1
            min_cell_x = int((min_x1 - safety_margin) // cell_size)
            min_cell_y = int((min_y1 - safety_margin) // cell_size)
            max_cell_x = int((max_x1 + safety_margin) // cell_size)
            max_cell_y = int((max_y1 + safety_margin) // cell_size)

            # Look in all neighboring grid cells for potential “c2” partners
            for cx in range(min_cell_x, max_cell_x + 1):
                for cy in range(min_cell_y, max_cell_y + 1):
                    for c2_idx in spatial_grid[(cx, cy)]:
                        if c2_idx <= idx or c2_idx in merged_indices:
                            continue
                        # Check actual overlap between bbox1 and bbox2
                        min_x2, min_y2, max_x2, max_y2 = prev_bboxes[c2_idx]
                        if not (max_x1 < min_x2 or max_x2 < min_x1 
                                or max_y1 < min_y2 or max_y2 < min_y1):
                            # They overlap—so merge c2 into c1
                            row1 = refined_df.loc[idx]
                            row2 = refined_df.loc[c2_idx]
                            new_x = np.concatenate([row1['x_coords'], row2['x_coords']])
                            new_y = np.concatenate([row1['y_coords'], row2['y_coords']])
                            new_T = np.concatenate([row1['T_IR'],    row2['T_IR']])

                            # Deduplicate coordinates
                            coords = np.unique(np.column_stack((new_x, new_y)), axis=0)
                            refined_df.at[idx, 'x_coords'] = coords[:, 0].astype(np.uint32)
                            refined_df.at[idx, 'y_coords'] = coords[:, 1].astype(np.uint32)
                            refined_df.at[idx, 'size']  = len(coords)
                            refined_df.at[idx, 'T_IR']  = new_T[:len(coords)].astype(np.float32)

                            # Mark c2 for dropping
                            merged_indices.add(c2_idx)

        # Finally, drop all merged indices—**both** from refined_df and from edge_mask
        if merged_indices:
            refined_df.drop(list(merged_indices), inplace=True)
            edge_mask.drop(list(merged_indices), inplace=True)
    # end of timestamp loop

    elapsed = time.time() - start_time
    print(f"Dateline reconciliation completed in {elapsed:.1f} s; final row count = {len(refined_df)}")
    return refined_df


#############################
# SEQUENTIAL RENUMBERING FUNCTIONS
#############################

def renumber_clusters_sequential(df, offset=0):
    """
    Renumber clusters sequentially, applying an offset to ensure global uniqueness across chunks.
    Returns the renumbered DataFrame and the next offset.
    """
    if df.empty:
        return df, offset

    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    new_ids = pd.Series(range(offset + 1, offset + len(df_sorted) + 1), dtype=np.uint32)
    df_sorted['cluster_id'] = new_ids
    
    new_offset = offset + len(df_sorted)
    return df_sorted, new_offset

#############################
# TEMPORAL RECONCILIATION FUNCTIONS
#############################

def reconcile_single_step(current_df, prev_df, overlap_percentage):
    """Helper function to reconcile one timestep against a previous one."""
    if prev_df.empty or current_df.empty:
        return current_df, prev_df

    current_clusters = {row.cluster_id: row for _, row in current_df.iterrows()}
    prev_clusters = {row.cluster_id: row for _, row in prev_df.iterrows()}
    
    prev_coords = {pid: set(zip(p.x_coords, p.y_coords)) for pid, p in prev_clusters.items()}

    current_to_prev_best_match = {}
    
    # For each current cluster, find the best previous cluster it overlaps with
    for cid, c in current_clusters.items():
        current_set = set(zip(c.x_coords, c.y_coords))
        best_prev_id = None
        max_overlap_metric = -1 # Size for simple overlap, percentage for strict
        
        for pid, p_set in prev_coords.items():
            intersection_size = len(current_set.intersection(p_set))
            if intersection_size == 0:
                continue
                
            if overlap_percentage < 0: # Any overlap, choose largest previous storm
                overlap_metric = prev_clusters[pid]['size']
            else: # Percentage based
                overlap_metric = intersection_size / prev_clusters[pid]['size']
            
            if overlap_metric >= overlap_percentage and overlap_metric > max_overlap_metric:
                max_overlap_metric = overlap_metric
                best_prev_id = pid
        
        if best_prev_id:
            current_to_prev_best_match[cid] = best_prev_id

    # --- Handle splits and mergers ---
    prev_to_current_map = defaultdict(list)
    for cid, pid in current_to_prev_best_match.items():
        prev_to_current_map[pid].append(cid)
    
    reconciled_rows = []
    processed_cids = set()

    for pid, cids in prev_to_current_map.items():
        # This group of current clusters (cids) all matched to the same previous cluster (pid)
        # This handles both one-to-one continuation and splits (one prev -> many current)
        for cid in cids:
            reconciled_row = current_clusters[cid].copy()
            reconciled_row['cluster_id'] = pid # Inherit the ID
            reconciled_rows.append(reconciled_row)
            processed_cids.add(cid)
    
    # Handle mergers: if multiple previous clusters map to the same current cluster
    current_to_prev_map = defaultdict(list)
    for cid, pid in current_to_prev_best_match.items():
        current_to_prev_map[cid].append(pid)
        
    for cid, pids in current_to_prev_map.items():
        if len(pids) > 1:
            # A merge happened. The current cluster (cid) already inherited the ID of the largest previous storm.
            # We just need to mark the smaller previous storms as 'merged'.
            largest_prev_id = current_to_prev_best_match[cid]
            for pid in pids:
                if pid != largest_prev_id:
                    # Find this cluster in the previous dataframe and mark it
                    prev_df.loc[prev_df['cluster_id'] == pid, 'merged'] = True
                    prev_df.loc[prev_df['cluster_id'] == pid, 'merged_into'] = largest_prev_id

    # Add any new storms (current clusters that didn't match any previous ones)
    for cid, c in current_clusters.items():
        if cid not in processed_cids:
            reconciled_rows.append(c)

    return pd.DataFrame(reconciled_rows), prev_df


def reconcile_temporal_clusters_chunk_aware(df_renumbered, prev_chunk_last_df=None, overlap_percentage=-1):
    """
    Performs temporal reconciliation on an in-memory DataFrame chunk,
    and can use the last timestep of a previous chunk for continuity.
    """
    if df_renumbered.empty:
        return df_renumbered
        
    df = df_renumbered.sort_values('timestamp').reset_index(drop=True)
    df['merged'] = False
    df['merged_into'] = pd.Series(dtype='UInt32') # Use nullable integer type

    timestamps = sorted(df['timestamp'].unique())
    reconciled_dfs = []
    
    # --- Step 1: Handle the very first timestep of the chunk ---
    first_ts_df = df[df['timestamp'] == timestamps[0]].copy()
    prev_df = prev_chunk_last_df
    
    if prev_df is None:
        # This is the first chunk overall, no reconciliation needed for its first step
        reconciled_dfs.append(first_ts_df)
        prev_df = first_ts_df
    else:
        # Reconcile the first step of this chunk against the last step of the previous chunk
        print("  Stitching to previous chunk...")
        reconciled_first_step_df, _ = reconcile_single_step(first_ts_df, prev_df, overlap_percentage)
        reconciled_dfs.append(reconciled_first_step_df)
        prev_df = reconciled_first_step_df
        
    # --- Step 2: Loop through the rest of the timesteps in the chunk ---
    for i in tqdm(range(1, len(timestamps)), desc="  Temporal Reconciliation", leave=False):
        current_df = df[df['timestamp'] == timestamps[i]].copy()
        reconciled_step_df, updated_prev_df_merges = reconcile_single_step(current_df, prev_df, overlap_percentage)
        
        # Update the 'merged' status in the main list of dataframes
        if not updated_prev_df_merges.empty:
            prev_ts = timestamps[i-1]
            # Find the index of the df to update
            for idx, df_to_update in enumerate(reconciled_dfs):
                if not df_to_update.empty and df_to_update['timestamp'].iloc[0] == prev_ts:
                    reconciled_dfs[idx] = updated_prev_df_merges
                    break

        reconciled_dfs.append(reconciled_step_df)
        prev_df = reconciled_step_df

    return pd.concat(reconciled_dfs, ignore_index=True)


#############################
# MAIN EXECUTION
#############################

def format_coordinates(lat_array, lon_array):
    # Reverse latitude array to match the function output format
    #lat = np.array(lat_array)[::-1]
    lat = np.array(lat_array)
    lon = np.array(lon_array)
    return lat, lon

def run_storm_tracking_pipeline_for_tos2ca(
    images, timestamps, grid, temp_thresh=245, temp_warmer_thresh=265, min_size=81,
    max_size_threshold=2500, overlap_percentage=-1, toggle="on"
):
    """
    Run the AUX-GEOIR code

    Parameters
    ----------
    images : ODict
        Ordered dictionary of input data
    timestamps: list
        Array of timestamps
    grid: fortracc_module.objects.SparseGeoGrid
        Spatial grid
    temp_thresh: int
        Temperature threshold in Kelvin
    temp_warmer_thresh: int
        Secondary warmer temperature threshold
    min_size: int
        Minimum cluster size to retain (in pixels)
    max_size_threshold: int
        Maximum cluster size threshold
    overlap_percentage: int
        Overlap percentage for temporal reconciliation (negative for any overlap)
    toggle: str
        Toggle for clustering approach: "on" for two-stage, "off" for one-stage


    Returns
    -------
    sparse_seq: fortracc_module.flow.SparseTimeOrderedSequence
        Storm information
    """
    if len(images) != len(timestamps):
        raise ValueError(
            "The number of images must be equal to the number of timestamps." +
            f" Got {len(images)} images and {len(timestamps)} timestamps."
        )
    for image in images:
        if image.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays for masks.  Got {image.ndim}D array instead."
            )

    for t in timestamps:
        if len(t) != 12:
            raise ValueError(
                "Expected all timestamps to be formatted as YYYYMMDDhhmm(e.g. 201501011430)."
            )
    print("\n" + "="*80)
    print("BEGINNING OPTIMIZED STORM TRACKING PIPELINE (CHUNK-BASED)")
    print(f"Configuration: chunk_size={len(images)}")   
    print("="*80)


    # --- Step 1: Initial Setup and File Planning ---
    print("\nPlanning processing chunks...")
    lat_array, lon_array = format_coordinates(grid.latitude, grid.longitude)
    lat_array_f32 = lat_array.astype(np.float32)
    lon_array_f32 = lon_array.astype(np.float32)

    tasks = []
    for i in range(len(images)):
        file_index = i
        timestep_index = i
        tasks.append((images[file_index], timestep_index))

    timesteps_per_chunk = len(tasks)
    chunked_tasks = [tasks[i:i + timesteps_per_chunk] for i in range(0, len(tasks), timesteps_per_chunk)]
    print(f"Total timesteps in this chunk: {len(tasks)}.")

    # --- Step 2: Main Loop - Process Data in Chunks ---
    cluster_id_offset = 0
    last_df_of_prev_chunk = None


    for i, chunk_tasks in enumerate(chunked_tasks):    
        # --- Stage A: Initial Clustering (In-Memory) ---
        print(f"Step 1: Initial Clustering for {len(chunk_tasks)} timesteps...")
        worker_args = [
            (
            chunk_tasks[i][0], # image 1
            chunk_tasks[i][1], # ts_idx 1
            timestamps[chunk_tasks[i][1]], # timestamp 1

            chunk_tasks[i + 1][0] if i + 1 < len(chunk_tasks) else None, # image 1 if exists, none otherwise
            chunk_tasks[i + 1][1] if i + 1 < len(chunk_tasks) else None, # ts_idx 2 if exists, none otherwise
            timestamps[chunk_tasks[i + 1][1]] if i + 1 < len(chunk_tasks) else None,

            temp_thresh, temp_warmer_thresh, min_size, max_size_threshold, 
            toggle if i + 1 < len(chunk_tasks) else "off")     # toggle: the input value of "toggle" if I have a next measurement; "off" otherwise (for the last task)
            for i in range(len(chunk_tasks))
        ]

        all_chunk_dfs = []
        all_chunk_dfs = [worker_process_in_memory_for_tos2ca(arg) for arg in tqdm(worker_args, desc="  Initial Clustering")]

        valid_dfs = [
            df.dropna(axis=1, how='all')  # drop all-NA columns
            for df in all_chunk_dfs
            if not df.empty and not df.isna().all(axis=None)
        ]
        if valid_dfs:
            chunk_df_initial = pd.concat(valid_dfs, ignore_index=True)
        else:
            chunk_df_initial = pd.DataFrame()        
        
        if chunk_df_initial.empty:
            print("  No clusters found in this chunk. Skipping.")
            continue                               
        
        # --- Stage B: Dateline Reconciliation ---
        print("Step 2: Dateline Reconciliation...")
        chunk_df_dateline = reconcile_dateline_clusters(chunk_df_initial, lon_array_f32, lat_array_f32)
        del chunk_df_initial
        gc.collect()

        # --- Stage C: Sequential Renumbering (with offset) ---
        print("Step 3: Sequential Renumbering...")
        chunk_df_renumbered, cluster_id_offset = renumber_clusters_sequential(
            chunk_df_dateline, offset=cluster_id_offset
        )
        print(f"  New cluster ID offset: {cluster_id_offset}")
        del chunk_df_dateline
        gc.collect()
        
        # --- Stage D: Temporal Reconciliation (Chunk-Aware) ---
        print("Step 4: Temporal Reconciliation...")
        chunk_df_temporal = reconcile_temporal_clusters_chunk_aware(
            chunk_df_renumbered,
            prev_chunk_last_df=last_df_of_prev_chunk,
            overlap_percentage=overlap_percentage
        )
        del chunk_df_renumbered
        gc.collect()
        
        # --- Stage E: Pass-through for Unfiltered Data ---
        print("Step 5: Deferring filtering to a final, global step.")
        chunk_df_unfiltered = chunk_df_temporal.copy() # Use a new variable for clarity
        del chunk_df_temporal
        gc.collect()
        
        print("Step 6: Appending raw tracked data (with x/y coords) to unfiltered output file...")
        if not chunk_df_unfiltered.empty:
    
            # Define the columns to keep. We now keep x/y coords and drop lons/lats.
            final_columns = ['cluster_id', 'x_coords', 'y_coords', 'T_IR', 'timestamp', 'size', 
                     'merged', 'merged_into', 'min_temp', 'mean_temp']
    
            # Ensure only existing columns are selected
            chunk_to_save = chunk_df_unfiltered[[col for col in final_columns if col in chunk_df_unfiltered.columns]]

            df_filtered = filter_storms_in_memory(chunk_to_save, temp_thresh=temp_thresh, max_size_threshold=max_size_threshold, min_duration_steps=6)

            sparse_seq = convert_df_to_sparse_sequence(df=df_filtered, timestamps=timestamps, grid=grid, cls=SparseTimeOrderedSequence)

    return sparse_seq