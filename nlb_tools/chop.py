import os
import h5py
import logging
from os import path
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class SegmentRecord:
    """Stores information needed to reconstruct a segment from chops"""
    def __init__(self, seg_id, clock_time, offset, n_chops, overlap):
        """Stores the information needed to reconstruct a segment

        Parameters
        ----------
        seg_id : int
            The ID of this segment.
        clock_time : pd.Series
            The TimeDeltaIndex of the original data from this segment
        offset : int
            The offset of the chops from the start of the segment
        n_chops : int
            The number of chops that make up this segment
        overlap : int
            The number of bins of overlap between adjacent chops
        """
        self.seg_id = seg_id
        self.clock_time = clock_time
        self.offset = offset
        self.n_chops = n_chops
        self.overlap = overlap

    def rebuild_segment(self, chops, smooth_pwr=2):
        """Reassembles a segment from its chops

        Parameters
        ----------
        chops : np.ndarray
            A 3D numpy array of shape n_chops x seg_len x data_dim that 
            holds the data from all of the chops in this segment
        smooth_pwr : float, optional
            The power to use for smoothing. See `merge_chops` 
            function for more details, by default 2

        Returns
        -------
        pd.DataFrame
            A DataFrame of reconstructed segment data, indexed by the 
            clock_time of the original segment
        """
        # Merge the chops for this segment
        merged_array = merge_chops(
            chops, 
            overlap=self.overlap, 
            orig_len=len(self.clock_time) - self.offset,
            smooth_pwr=smooth_pwr)
        # Add NaNs for points that were not modeled due to offset
        data_dim = merged_array.shape[1]
        offset_nans = np.full((self.offset, data_dim), np.nan)
        merged_array = np.concatenate([offset_nans, merged_array])
        # Recreate segment DataFrame with the appropriate `clock_time`s
        try:
            segment_df = pd.DataFrame(merged_array, index=self.clock_time)
        except:
            import pdb; pdb.set_trace()
            segment_df = None
        return segment_df

class ChopInterface:
    """Chops data from NWBDatasets into segments with fixed overlap"""
    def __init__(self,
                 window,
                 overlap,
                 max_offset=0,
                 chop_margins=0,
                 random_seed=None):
        """Initializes a ChopInterface

        Parameters
        ----------
        window : int
            The length of chopped segments in ms
        overlap : int
            The overlap between chopped segments in ms
        max_offset : int, optional
            The maximum offset of the first chop from the beginning of 
            each segment in ms. The actual offset will be chose 
            randomly. By default, 0 adds no offset
        chop_margins : int, optional
            The size of extra margins to add to either end of each chop 
            in bins, designed for use with the temporal_shift operation,
            by default 0
        random_seed : int, optional
            The random seed for generating the dataset, by default None
            does not use a random seed
        """
        def to_timedelta(ms):
            return pd.to_timedelta(ms, unit='ms')

        self.window = to_timedelta(window)
        self.overlap = to_timedelta(overlap)
        self.max_offset = to_timedelta(max_offset)
        self.chop_margins = chop_margins
        self.random_seed = random_seed

    def chop(self, neural_df, chop_fields):
        """Chops a trialized or continuous RDS DataFrame.

        Parameters
        ----------
        neural_df : pd.DataFrame
            A continuous or trialized DataFrame from RDS.
        chop_fields : str or list of str
            `signal_type` or list of `signal_type`s in neural_df to chop

        Returns
        -------
        dict of np.array
            A data_dict of the chopped data. Consists of a dictionary 
            with data tags mapping to 3D numpy arrays with dimensions 
            corresponding to samples x time x features.
        """
        # Set the random seed for the offset
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        if type(chop_fields) != list:
            chop_fields = [chop_fields]
        
        # Get info about the column groups to be chopped
        data_fields = sorted(chop_fields)
        get_field_dim = lambda field: getattr(neural_df, field).shape[1] if len(getattr(neural_df, field).shape) > 1 else 1
        data_dims = [get_field_dim(f) for f in data_fields]
        data_splits = np.cumsum(data_dims[:-1])
        # Report information about the fields that are being chopped
        logger.info(f'Chopping data field(s) {data_fields} with dimension(s) {data_dims}.')
        
        # Calculate bin widths and set up segments for chopping
        if 'trial_id' in neural_df:
            # Trialized data
            bin_width = neural_df.clock_time[1] - neural_df.clock_time[0]
            segments = neural_df.groupby('trial_id')
        else:
            # Continuous data
            bin_width = neural_df.index[1] - neural_df.index[0]
            if np.any(np.isnan(neural_df[chop_fields])):
                splits = np.where(neural_df[chop_fields].sum(axis=1, min_count=1).isna().diff())[0].tolist() + [len(neural_df)]
                segments = {n: neural_df.iloc[splits[n]:splits[n+1]].reset_index() for n in range(0, len(splits) - 1, 2)}
            else:
                segments = {1: neural_df.reset_index()}.items()

        # Calculate the number of bins to use for chopping parameters
        window = int(self.window / bin_width)
        overlap = int(self.overlap / bin_width)
        chop_margins_td = pd.to_timedelta(
            self.chop_margins * bin_width, unit='ms')

        # Get correct offset based on data type
        if 'trial_id' in neural_df:
            # Trialized data
            max_offset = int(self.max_offset / bin_width)
            max_offset_td = self.max_offset
            get_offset = lambda: np.random.randint(max_offset+1)
        else:
            # Continuous data
            max_offset = 0
            max_offset_td = pd.to_timedelta(max_offset)
            get_offset = lambda: 0
            if self.max_offset > pd.to_timedelta(0):
                # Doesn't make sense to use offset on continuous data
                logger.info("Ignoring offset for continuous data.")

        def to_ms(timedelta):
            return int(timedelta.total_seconds() * 1000)

        # Log information about the chopping to be performed
        chop_message = ' - '.join([
            'Chopping data',
            f'Window: {window} bins, {to_ms(self.window)} ms',
            f'Overlap: {overlap} bins, {to_ms(self.overlap)} ms',
            f'Max offset: {max_offset} bins, {to_ms(max_offset_td)} ms',
            f'Chop margins: {self.chop_margins} bins, {to_ms(chop_margins_td)} ms',
        ])
        logger.info(chop_message)

        # Iterate through segments, which can be trials or continuous data
        data_dict = defaultdict(list)
        segment_records = []
        for segment_id, segment_df in segments:
            # Get the data from all of the column groups to extract
            data_arrays = [getattr(segment_df, f).to_numpy() if len(getattr(segment_df, f).shape) > 1 
                           else getattr(segment_df, f).to_numpy()[:, None] for f in data_fields]
            # Concatenate all data types into a single segment array
            segment_array = np.concatenate(data_arrays, axis=1)
            if self.chop_margins > 0:
                # Add padding to segment if we are using chop margins
                seg_dim = segment_array.shape[1]
                pad = np.full((self.chop_margins, seg_dim), 0.0001)
                segment_array = np.concatenate([pad, segment_array, pad])
            # Sample an offset for this segment
            offset = get_offset()
            # Chop all of the data in this segment
            chops = chop_data(
                segment_array, 
                overlap + 2*self.chop_margins, 
                window + 2*self.chop_margins, 
                offset)
            # Split the chops back up into the original fields
            data_chops = np.split(chops, data_splits, axis=2)
            # Create the data_dict with LFADS input names
            for field, data_chop in zip(data_fields, data_chops):
                data_dict[field].append(data_chop)
            # Keep a record to represent each original segment
            seg_rec = SegmentRecord(
                segment_id,
                segment_df.clock_time,
                offset,
                len(chops),
                overlap)
            segment_records.append(seg_rec)
        # Store the information for reassembling segments
        self.segment_records = segment_records
        # Consolidate data from all segments into a single array
        data_dict = {name: np.concatenate(c) for name, c in data_dict.items()}
        # Report diagnostic info
        dict_key = list(data_dict.keys())[0]
        n_chops = len(data_dict[dict_key])
        n_segments = len(segment_records)
        logger.info(f'Created {n_chops} chops from {n_segments} segment(s).')

        return data_dict

    def merge(self, chopped_data, smooth_pwr=2):
        """Merges chopped data to reconstruct the original input 
        sequence

        Parameters
        ----------
        chopped_data : dict
            Dict mapping the keys to chopped 3d numpy arrays
        smooth_pwr : float, optional
            The power to use for smoothing. See `merge_chops` 
            function for more details, by default 2

        Returns
        -------
        pd.DataFrame
            A merged DataFrame indexed by the clock time of the original 
            chops. Columns are multiindexed using `fields_map`. 
            Unmodeled data is indicated by NaNs.
        """
        # Get the desired arrays from the output
        output_fields = sorted(chopped_data.keys())
        output_arrays = [chopped_data[f] for f in output_fields]
        # Keep track of boundaries between the different signals
        output_dims = [a.shape[-1] for a in output_arrays]
        # Concatenate the output arrays for more efficient merging
        output_full = np.concatenate(output_arrays, axis=-1)
        # Get info for separating the chops related to each segment
        seg_splits = np.cumsum([s.n_chops for s in self.segment_records])[:-1]
        # Separate out the chops for each segment
        seg_chops = np.split(output_full, seg_splits, axis=0)
        # Reconstruct the segment DataFrames
        segment_dfs = [record.rebuild_segment(chops, smooth_pwr) \
            for record, chops in zip(self.segment_records, seg_chops)]
        # Concatenate the segments with clock_time indices
        merged_df = pd.concat(segment_dfs)
        # Add multiindexed columns
        midx_tuples = [(sig, f'{i:04}') \
            for sig, dim in zip(output_fields, output_dims) \
                for i in range(dim)]
        merged_df.columns = pd.MultiIndex.from_tuples(midx_tuples)

        return merged_df


def chop_data(data, overlap, window, offset=0):
    """Rearranges an array of continuous data into overlapping segments. 
    
    This low-level function takes a 2-D array of features measured 
    continuously through time and breaks it up into a 3-D array of 
    partially overlapping time segments.

    Parameters
    ----------
    data : np.ndarray
        A TxN numpy array of N features measured across T time points.
    overlap : int
        The number of points to overlap between subsequent segments.
    window : int
        The number of time points in each segment.
    Returns
    -------
    np.ndarray
        An SxTxN numpy array of S overlapping segments spanning 
        T time points with N features.
    
    See Also
    --------
    chop.merge_chops : Performs the opposite of this operation.
    """
    # Random offset breaks temporal connection between trials and chops
    offset_data = data[offset:]
    shape = (
        int((offset_data.shape[0] - window)/(window - overlap)) + 1, 
        window,
        offset_data.shape[-1],
    )
    strides = (
        offset_data.strides[0]*(window - overlap), 
        offset_data.strides[0], 
        offset_data.strides[1],
    )
    chopped = np.lib.stride_tricks.as_strided(
        offset_data, shape=shape, strides=strides).copy().astype('f')
    return chopped


def merge_chops(data, overlap, orig_len=None, smooth_pwr=2):
    """Merges an array of overlapping segments back into continuous data.
    This low-level function takes a 3-D array of partially overlapping 
    time segments and merges it back into a 2-D array of features measured
    continuously through time.

    Parameters
    ----------
    data : np.ndarray
        An SxTxN numpy array of S overlapping segments spanning 
        T time points with N features.
    overlap : int
        The number of overlapping points between subsequent segments.
    orig_len : int, optional
        The original length of the continuous data, by default None
        will cause the length to depend on the input data.
    smooth_pwr : float, optional
        The power of smoothing. To keep only the ends of chops and 
        discard the beginnings, use np.inf. To linearly blend the 
        chops, use 1. Raising above 1 will increasingly prefer the 
        ends of chops and lowering towards 0 will increasingly 
        prefer the beginnings of chops (not recommended). To use 
        only the beginnings of chops, use 0 (not recommended). By 
        default, 2 slightly prefers the ends of segments.
    Returns
    -------
    np.ndarray
        A TxN numpy array of N features measured across T time points.
    
    See Also
    --------
    chop.chop_data : Performs the opposite of this operation.
    """ 
    if smooth_pwr < 1:
        logger.warning('Using `smooth_pwr` < 1 for merging '
            'chops is not recommended.')

    merged = []
    full_weight_len = data.shape[1] - 2*overlap
    # Create x-values for the ramp
    x = np.linspace(1/overlap, 1 - 1/overlap, overlap) \
        if overlap != 0 else np.array([])
    # Compute a power-function ramp to transition
    ramp = 1 - x ** smooth_pwr
    ramp = np.expand_dims(ramp, axis=-1)
    # Compute the indices to split up each chop
    split_ixs = np.cumsum([overlap, full_weight_len])
    for i in range(len(data)):
        # Split the chop into overlapping and non-overlapping
        first, middle, last = np.split(data[i], split_ixs)
        # Ramp each chop and combine it with the previous chop
        if i == 0:
            last = last * ramp
        elif i == len(data) - 1:
            first = first * (1-ramp) + merged.pop(-1)
        else:
            first = first * (1-ramp) + merged.pop(-1)
            last = last * ramp
        # Track all of the chops in a list
        merged.extend([first, middle, last])
    
    merged = np.concatenate(merged)
    # Indicate unmodeled data with NaNs
    if orig_len is not None and len(merged) < orig_len:
        nans = np.full((orig_len-len(merged), merged.shape[1]), np.nan)
        merged = np.concatenate([merged, nans])
    
    return merged