
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.core import MultiContainerInterface, NWBDataInterface
from scipy.stats import mode
from glob import glob
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.interpolate as interpolate
import multiprocessing
import itertools
import os
import logging

logger = logging.getLogger(__name__)


class NWBDataset:
    """A class for loading/preprocessing data from NWB files for
    the NLB competition
    """
    def __init__(self, fpath, prefix='', split_heldout=True, skip_fields=[]):
        """Initializes an NWBDataset, loading data from 
        the indicated file(s)

        Parameters
        ----------
        fpath : str
            Either the path to an NWB file or to a directory
            containing NWB files
        prefix : str, optional
            A pattern used to filter the NWB files in directory
            by name. By default, prefix='' loads all .nwb files in
            the directory. Please refer to documentation for
            the `glob` module for more details: 
            https://docs.python.org/3/library/glob.html
        split_heldout : bool, optional
            Whether to load heldin units and heldout units
            to separate fields or not, by default True
        skip_fields : list, optional
            List of field names to skip during loading,
            which may be useful if memory is an issue.
            Field names must match the names automatically
            assigned in the loading process. Spiking data 
            can not be skipped. Field names in the list
            that are not found in the dataset are
            ignored
        """
        fpath = os.path.expanduser(fpath)
        self.fpath = fpath
        self.prefix = prefix
        # Check if file/directory exists
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Specified file or directory not found")
        # If directory, look for files with matching prefix
        if os.path.isdir(fpath):
            filenames = sorted(glob(os.path.join(fpath, prefix + "*.nwb")))
        else:
            filenames = [fpath]
        # If no files found
        if len(filenames) == 0:
            raise FileNotFoundError(f"No matching files with prefix {prefix} found in directory {fpath}")
        # If multiple files found
        elif len(filenames) > 1:
            loaded = [self.load(fname, split_heldout=split_heldout, skip_fields=skip_fields) for fname in filenames]
            datas, trial_infos, descriptions, bin_widths = [list(out) for out in zip(*loaded)]
            assert np.all(np.array(bin_widths) == bin_widths[0]), "Bin widths of loaded datasets must be the same"
            # Shift loaded files to stack them into continuous array
            def trial_shift(x, shift_ms, trial_offset):
                if x.name.endswith('_time'):
                    return x + pd.to_timedelta(shift_ms, unit='ms')
                elif x.name == 'trial_id':
                    return x + trial_offset
                else:
                    return x
            # Loop through files, shifting continuous data
            past_end = datas[0].index[-1].total_seconds() + round(50 * bin_widths[0] / 1000, 4)
            descriptions_full = descriptions[0]
            tcount = len(trial_infos[0])
            for i in range(1, len(datas)):
                block_start_ms = np.ceil(past_end * 10) * 100
                datas[i] = datas[i].shift(block_start_ms, freq='ms')
                trial_infos[i] = trial_infos[i].apply(trial_shift, shift_ms=block_start_ms, trial_offset=tcount)
                descriptions_full.update(descriptions[i])
                past_end = datas[i].index[-1].total_seconds() + round(50 * bin_widths[i] / 1000, 4)
                tcount += len(trial_infos[i])
            # Stack data and reindex to continuous
            self.data = pd.concat(datas, axis=0, join='outer')
            self.trial_info = pd.concat(trial_infos, axis=0, join='outer').reset_index(drop=True)
            self.descriptions = descriptions_full
            self.bin_width = bin_widths[0]
            new_index = pd.to_timedelta((np.arange(round(self.data.index[-1].total_seconds() * 1000 / self.bin_width) + 1) * self.bin_width).round(4), unit='ms')
            self.data = self.data.reindex(new_index)
            self.data.index.name = 'clock_time'
        # If single file found
        else:
            data, trial_info, descriptions, bin_width = self.load(filenames[0], split_heldout=split_heldout, skip_fields=skip_fields)
            self.data = data
            self.trial_info = trial_info
            self.descriptions = descriptions
            self.bin_width = bin_width
        
    def load(self, fpath, split_heldout=True, skip_fields=[]):
        """Loads data from an NWB file into two dataframes,
        one for trial info and one for time-varying data

        Parameters
        ----------
        fpath : str
            Path to the NWB file
        split_heldout : bool, optional
            Whether to load heldin units and heldout units
            to separate fields or not, by default True
        skip_fields : list, optional
            List of field names to skip during loading,
            which may be useful if memory is an issue.
            Field names must match the names automatically
            assigned in the loading process. Spiking data 
            can not be skipped. Field names in the list
            that are not found in the dataset are
            ignored

        Returns
        -------
        tuple
            Tuple containing a pd.DataFrame of continuous loaded
            data, a pd.DataFrame with trial metadata, a dict
            with descriptions of fields in the DataFrames, and
            the bin width of the loaded data in ms
        """
        logger.info(f"Loading {fpath}")

        # Open NWB file
        io = NWBHDF5IO(fpath, 'r')
        nwbfile = io.read()

        # Load trial info and units
        trial_info = (
            nwbfile.trials.to_dataframe()
            .reset_index()
            .rename({'id': 'trial_id', 'stop_time': 'end_time'}, axis=1))
        units = nwbfile.units.to_dataframe()

        # Load descriptions of trial info fields
        descriptions = {}
        for name in nwbfile.trials.colnames:
            if not hasattr(nwbfile.trials, name):
                logger.warning(f"Field {name} not found in NWB file trials table")
                continue
            descriptions[name] = getattr(nwbfile.trials, name).description

        # Find all timeseries
        def make_df(ts):
            """Converts TimeSeries into pandas DataFrame"""
            if ts.timestamps is not None:
                index = ts.timestamps[()]
            else:
                index = np.arange(ts.data.shape[0]) / ts.rate + ts.starting_time
            columns = ts.comments.split('[')[-1].split(']')[0].split(',') if 'columns=' in ts.comments else None
            df = pd.DataFrame(ts.data[()], index=pd.to_timedelta(index, unit='s'), columns=columns)
            return df

        def find_timeseries(nwbobj):
            """Recursively searches the NWB file for time series data"""
            ts_dict = {}
            for child in nwbobj.children:
                if isinstance(child, TimeSeries):
                    if child.name in skip_fields:
                        continue
                    ts_dict[child.name] = make_df(child)
                    descriptions[child.name] = child.description
                elif isinstance(child, ProcessingModule):
                    pm_dict = find_timeseries(child)
                    ts_dict.update(pm_dict)
                elif isinstance(child, MultiContainerInterface):
                    for field in child.children:
                        if isinstance(field, TimeSeries):
                            name = child.name + "_" + field.name
                            if name in skip_fields:
                                continue
                            ts_dict[name] = make_df(field)
                            descriptions[name] = field.description
            return ts_dict

        # Create a dictionary containing DataFrames for all time series
        data_dict = find_timeseries(nwbfile)

        # Calculate data index
        start_time = 0.0
        bin_width = 1 # in ms, this will be the case for all provided datasets
        rate = round(1000. / bin_width, 2) # in Hz
        # Use obs_intervals, or last trial to determine data end
        end_time = round(max(units.obs_intervals.apply(lambda x: x[-1][-1])) * rate) * bin_width
        if (end_time < trial_info['end_time'].iloc[-1]):
            print("obs_interval ends before trial end") # TO REMOVE
            end_time = round(trial_info['end_time'].iloc[-1] * rate) * bin_width
        timestamps = (np.arange(start_time, end_time, bin_width) / 1000).round(6)
        timestamps_td = pd.to_timedelta(timestamps, unit='s')

        # Check that all timeseries match with calculated timestamps
        for key, val in list(data_dict.items()):
            if not np.all(np.isin(np.round(val.index.total_seconds(), 6), timestamps)):
                logger.warning(f"Dropping {key} due to timestamp mismatch.")
                data_dict.pop(key)

        def make_mask(obs_intervals):
            """Creates boolean mask to indicate when spiking data is not in obs_intervals"""
            mask = np.full(timestamps.shape, True)
            for start, end in obs_intervals:
                start_idx = np.ceil(round((start - timestamps[0]) * rate, 6)).astype(int)
                end_idx = np.floor(round((end - timestamps[0]) * rate, 6)).astype(int)
                mask[start_idx:end_idx] = False
            return mask
    
        # Prepare variables for spike binning
        masks = [(~units.heldout).to_numpy(), units.heldout.to_numpy()] if split_heldout else [np.full(len(units), True)]

        for mask, name in zip(masks, ['spikes', 'heldout_spikes']):
            # Check if there are any units
            if not np.any(mask):
                continue
            
            # Allocate array to fill with spikes
            spike_arr = np.full((len(timestamps), np.sum(mask)), 0.0, dtype='float16')

            # Bin spikes using decimal truncation and np.unique - faster than np.histogram with same results
            for idx, (_, unit) in enumerate(units[mask].iterrows()):
                spike_idx, spike_cnt = np.unique(((unit.spike_times - timestamps[0]) * rate).round(6).astype(int), return_counts=True)
                spike_arr[spike_idx, idx] = spike_cnt

            # Replace invalid intervals in spike recordings with NaNs
            if 'obs_intervals' in units.columns:
                neur_mask = make_mask(units[mask].iloc[0].obs_intervals)
                if np.any(spike_arr[neur_mask]):
                    logger.warning("Spikes found outside of observed interval.")
                spike_arr[neur_mask] = np.nan

            # Create DataFrames with spike arrays
            data_dict[name] = pd.DataFrame(spike_arr, index=timestamps_td, columns=units[mask].index).astype('float16', copy=False)

        # Create MultiIndex column names
        data_list = []
        for key, val in data_dict.items():
            chan_names = None if type(val.columns) == pd.RangeIndex else val.columns
            val.columns = self._make_midx(key, chan_names=chan_names, num_channels=val.shape[1])
            data_list.append(val)
        
        # Assign time-varying data to `self.data`
        data = pd.concat(data_list, axis=1)
        data.index.name = 'clock_time'
        data.sort_index(axis=1, inplace=True)

        # Convert time fields in trial info to timedelta
        # and assign to `self.trial_info`
        def to_td(x):
            if x.name.endswith('_time'):
                return pd.to_timedelta(x, unit='s')
            else:
                return x
        trial_info = trial_info.apply(to_td, axis=0)

        io.close()

        return data, trial_info, descriptions, bin_width
    
    def make_trial_data(self,
                        start_field='start_time',
                        end_field='end_time', 
                        align_field=None, 
                        align_range=(None, None),
                        margin=0,
                        ignored_trials=None,
                        allow_overlap=False,
                        allow_nans=False):
        """Makes a DataFrame of trialized data based on 
        an alignment field

        Parameters
        ----------
        start_field : str, optional
            The field in `trial_info` to use as the beginning of
            each trial, by default 'start_time'
        end_field : str, optional
            The field in `trial_info` to use as the end of each trial,
            by default 'end_time'
        align_field : str, optional
            The field in `trial_info` to use for alignment,
            by default None, which does not align trials and
            instead takes them in their entirety
        align_range : tuple of int, optional
            The offsets to add to the alignment field to 
            calculate the alignment window, by default (None, None) 
            uses `trial_start` and `trial_end`
        margin : int, optional
            The number of ms of extra data to include on either end of 
            each trial, labeled with the `margin` column for easy 
            removal. Margins are useful for decoding and smoothing
        ignored_trials : pd.Series or np.ndarray, optional
            A boolean pd.Series or np.ndarray of the same length 
            as trial_info with True for the trials to ignore, by 
            default None ignores no trials. This is useful for 
            rejecting trials outside of the alignment process
        allow_overlap : bool, optional
            Whether to allow overlap between trials, by default False
            truncates each trial at the start of the subsequent trial
        allow_nans : bool, optional
            Whether to allow NaNs within trials, by default False
            drops all timestamps containing NaNs in any column
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing trialized data. It has the same
            fields as the continuous `self.data` DataFrame, but 
            adds `trial_id`, `trial_time`, and `align_time`. It also
            resets the index so `clock_time` is a column rather than
            an index. This DataFrame can be pivoted to plot its 
            various fields across trials, aligned relative to 
            `align_time`, `trial_time`, or `clock_time`
        """
        # Allow rejection of trials by passing a boolean series
        trial_info = self.trial_info.copy()
        trial_info['next_start'] = trial_info['start_time'].shift(-1)
        if ignored_trials is not None:
            trial_info = trial_info.loc[~ignored_trials]
        if len(trial_info) == 0:
            logger.warning("All trials ignored. No trial data made")
            return
        
        # Find alignment points
        bin_width = pd.to_timedelta(self.bin_width, unit='ms')
        trial_start = trial_info[start_field]
        trial_end = trial_info[end_field]
        next_start = trial_info['next_start']
        if align_field is not None:
            align_left = align_right = trial_info[align_field]
        else:
            align_field = f'{start_field} and {end_field}' # for logging
            align_left = trial_start
            align_right = trial_end
        
        # Find start and end points based on the alignment range
        start_offset, end_offset = pd.to_timedelta(align_range, unit='ms')
        if not pd.isnull(start_offset) and not pd.isnull(end_offset):
            if not ((end_offset - start_offset) / bin_width).is_integer():
                # Round align offsets if alignment range is not multiple of bin width
                end_offset = start_offset + (end_offset - start_offset).round(bin_width)
                align_range = (
                    int(round(start_offset.total_seconds() * 1000)),
                    int(round(end_offset.total_seconds() * 1000))
                )
                logger.warning('Alignment window not integer multiple of bin width. '
                    f'Rounded to {align_range}')
        if pd.isnull(start_offset):
            align_start = trial_start
        else:
            align_start = align_left + start_offset
        if pd.isnull(end_offset):
            # Subtract small interval to prevent inclusive timedelta .loc indexing
            align_end = trial_end - pd.to_timedelta(1, unit='us')
        else:
            align_end = align_right + end_offset - pd.to_timedelta(1, unit='us')

        # Add margins to either end of the data
        margin_delta = pd.to_timedelta(margin, unit='ms')
        margin_start = align_start - margin_delta
        margin_end = align_end + margin_delta

        trial_ids = trial_info['trial_id']

        # Store the alignment data in a dataframe
        align_data = pd.DataFrame({
            'trial_id': trial_ids,
            'margin_start': margin_start,
            'margin_end': margin_end,
            'align_start': align_start, 
            'align_end': align_end,
            'trial_start': trial_start,
            'align_left': align_left}).dropna()
        # Bound the end by the next trial / alignment start
        align_data['end_bound'] = (
            pd.concat([next_start, align_start], axis=1)
            .min(axis=1)
            .shift(-1))
        trial_dfs = []
        num_overlap_trials = 0
        def make_trial_df(args):
            idx, row = args
            # Handle overlap with the start of the next trial
            endpoint = row.margin_end
            trial_id = row.trial_id
            overlap = False
            if not pd.isnull(row.end_bound) and \
                row.align_end > row.end_bound:

                overlap = True
                if not allow_overlap:
                    # Allow overlapping margins, but not aligned data
                    endpoint = row.end_bound + margin_delta - pd.to_timedelta(1, unit='us')
            # Take a slice of the continuous data
            trial_idx = pd.Series(self.data.index[self.data.index.slice_indexer(row.margin_start, endpoint)])
            # Add trial identifiers
            trial_df = pd.DataFrame({
                ('trial_id', ''): np.repeat(trial_id, len(trial_idx)), 
                ('trial_time', ''): (trial_idx - row.trial_start.ceil(bin_width)),
                ('align_time', ''): (trial_idx - row.align_left.ceil(bin_width)),
                ('margin', ''): ((trial_idx < row.align_start) | (row.align_end < trial_idx))})
            trial_df.index = trial_idx
            return overlap, trial_df
        overlaps, trial_dfs = zip(*[make_trial_df(args) for args in align_data.iterrows()])
        num_overlap_trials = sum(overlaps)
        # Summarize alignment
        logger.info(f'Aligned {len(trial_dfs)} trials to '
            f'{align_field} with offset of {align_range} ms '
            f'and margin of {margin}.')
        # Report any overlapping trials to the user.
        if num_overlap_trials > 0:
            if allow_overlap:
                logger.warning(
                    f'Allowed {num_overlap_trials} overlapping trials.')
            else:
                logger.warning(
                    f'Shortened {num_overlap_trials} trials to prevent overlap.')
        # Combine all trials into one DataFrame
        trial_data = pd.concat(trial_dfs)
        trial_data.reset_index(inplace=True)
        trial_data = trial_data.merge(self.data, how='left', left_on=[('clock_time', '')], right_index=True)
        # Sanity check to make sure there are no duplicated `clock_time`'s
        if not allow_overlap:
            # Duplicated points in the margins are allowed
            td_nonmargin = trial_data[~trial_data.margin]
            assert td_nonmargin.clock_time.duplicated().sum() == 0, \
                'Duplicated points still found. Double-check overlap code.'
        # Make sure NaN's caused by adding trialized data to self.data are ignored
        nans_found = trial_data.isnull().sum().max()
        if nans_found > 0:
            pct_nan = (nans_found / len(trial_data)) * 100
            if allow_nans:
                logger.warning(f'NaNs found in {pct_nan:.2f}% of `trial_data`.')
            else:
                logger.warning(f'NaNs found in `self.data`. Dropping {pct_nan:.2f}% '
                    'of points to remove NaNs from `trial_data`.')
                trial_data = trial_data.dropna()
        trial_data.sort_index(axis=1, inplace=True)
        return trial_data

    def resample(self, target_bin):
        """Rebins spikes and performs antialiasing + downsampling on 
        continuous signals

        Parameters
        ----------
        target_bin : int
            The target bin size in milliseconds. Note that it must be an 
            integer multiple of self.bin_width
        """
        logger.info(f'Resampling data to {target_bin} ms.')
        # Check that resample_factor is an integer
        if target_bin == self.bin_width:
            logger.warning(f'Dataset already at {target_bin} ms resolution, skipping resampling...')
            return
        assert target_bin % self.bin_width == 0, \
            'target_bin must be an integer multiple of bin_width.'
        resample_factor = int(round(target_bin / self.bin_width))
       
        # Resample data based on signal type
        cols = self.data.columns
        data_list = []
        for signal_type in cols.get_level_values(0).unique():
            if 'spikes' in signal_type:
                # Rebin spikes, preserving original nan locations
                arr = self.data[signal_type].to_numpy()
                dtype = self.data[signal_type].dtypes.iloc[0]
                nan_mask = np.isnan(arr[::resample_factor])
                if arr.shape[0] % resample_factor != 0:
                    extra = arr[-(arr.shape[0] % resample_factor):]
                    arr = arr[:-(arr.shape[0] % resample_factor)]
                else:
                    extra = None
                arr = np.nan_to_num(arr, copy=False).reshape((arr.shape[0] // resample_factor, resample_factor, -1)).sum(axis=1)
                if extra is not None:
                    arr = np.vstack([arr, np.nan_to_num(extra, copy=False).sum(axis=0)])
                arr[nan_mask] = np.nan
                resamp = pd.DataFrame(arr, index=self.data.index[::resample_factor], dtype=dtype)
            elif signal_type == 'target_pos':
                # Resample target pos for MC_RTT
                resamp = self.data[signal_type].iloc[::resample_factor]
            else:
                # Resample with Chebyshev for other data types
                dtype = self.data[signal_type].dtypes.iloc[0]
                nan_mask = self.data[signal_type].iloc[::resample_factor].isna()
                if np.any(self.data[signal_type].isna()):
                    self.data[signal_type] = self.data[signal_type].apply(lambda x: x.interpolate(limit_direction='both'))
                decimated_df = signal.decimate(
                    self.data[signal_type], resample_factor, axis=0, n=500, ftype='fir')
                decimated_df[nan_mask] = np.nan
                resamp = pd.DataFrame(decimated_df, index=self.data.index[::resample_factor], dtype=dtype)
            resamp.columns = pd.MultiIndex.from_product([[signal_type], self.data[signal_type].columns], names=('signal_type', 'channel'))
            data_list.append(resamp)
        # Replace old data
        self.data = pd.concat(data_list, axis=1)
        self.data.index.freq = f'{target_bin}ms'
        self.bin_width = target_bin

    def smooth_spk(self, 
                   gauss_width, 
                   signal_type=None, 
                   name=None, 
                   overwrite=False,
                   ignore_nans=False,
                   parallelized=True,
                   dtype="float64"):
        """Applies Gaussian smoothing to the data. Most often
        applied to spikes
        
        Parameters
        ----------
        gauss_width : int
            The standard deviation of the Gaussian to use for
            smoothing, in ms
        signal_type : str or list of str, optional
            The group of signals to smooth, by default 
            None, which smooths 'spikes' and 'heldout_spikes'
            if present in the DataFrame
        name : str, optional
            The name to use for the smoothed data when adding 
            it back to the DataFrame, by default None. If
            provided, the new signal_type name will be 
            the original name + '_' + `name`. Must be provided
            if overwrite is False
        overwrite : bool, optional
            Whether to overwrite the original data,
            by default False
        ignore_nans : bool, optional
            Whether to ignore NaNs when smoothing, by default
            False. When NaNs are not ignored, they propagate
            into valid data during convolution, but ignoring
            NaNs is much slower
        parallelized : bool, optional
            Whether to parallelize the smoothing operation 
            with multiprocessing.Pool.map(). This may cause
            issues on certain systems, so it can be disabled
        dtype : str or dtype
            Data type for the smoothing output to be cast to,
            in case of memory issues or precision concerns.
            By default 'float64'. Only other float dtypes are
            recommended
        """
        assert name or overwrite, \
            ('You must either provide a name for the smoothed '
            'data or specify to overwrite the existing data.')
        
        if signal_type is None:
            signal_type = [field for field in ['spikes', 'heldout_spikes'] if field in self.data.columns]

        logger.info(f'Smoothing {signal_type} with a '
            f'{gauss_width} ms Gaussian.')

        # Compute Gauss window and std with respect to bins
        gauss_bin_std = gauss_width / self.bin_width
        # the window extends 3 x std in either direction
        win_len = int(6 * gauss_bin_std)
        # Create Gaussian kernel
        window = signal.gaussian(win_len, gauss_bin_std, sym=True)
        window /=  np.sum(window)
        # Extract spiking data
        spike_vals = self.data[signal_type].to_numpy()
        
        # Parallelized implementation for smoothing data
        if parallelized:
            spike_vals_list = [spike_vals[:,i] for i in range(spike_vals.shape[1])]
            y_list = _poolmap(
                smooth_column, itertools.product(spike_vals_list, [window], [ignore_nans], [dtype]))
            smoothed_spikes = np.vstack(y_list).T
        else:
            smoothed_spikes = np.apply_along_axis(lambda x: smooth_column((x, window, ignore_nans, dtype)), 0, spike_vals)

        # Create list of column names
        col_names = []
        if isinstance(signal_type, str):
            signal_type = [signal_type]
        for st in signal_type:
            columns = self.data[st].columns
            if overwrite:
                smoothed_name = st
            else:
                smoothed_name = st + '_' + name
            col_names += list(zip([smoothed_name]*len(columns), columns))

        # Write data to DataFrame
        if overwrite:
            self.data.drop(col_names, axis=1, inplace=True)
        smoothed_df = pd.DataFrame(smoothed_spikes, index=self.data.index, columns=pd.MultiIndex.from_tuples(col_names))
        self.data = pd.concat([self.data, smoothed_df], axis=1)
        self.data.sort_index(axis=1, inplace=True)
        # deleting and concatenating new data is much faster than overwriting, but less memory efficient
        # can replace with:
        # if overwrite:
        #     self.data[col_names] = smoothed_spikes
        # else:
        #     smoothed_df = pd.DataFrame(smoothed_spikes, index=self.data.index, columns=pd.MultiIndex.from_tuples(col_names))
        #     self.data = pd.concat([self.data, smoothed_df], axis=1)
        #     self.data.sort_index(axis=1, inplace=True)
        # if memory is an issue

    def add_continuous_data(self, cts_data, signal_type, chan_names=None):
        """Adds a continuous data field to the main DataFrame
        
        Parameters
        ----------
        cts_data : np.ndarray
            A numpy array whose first dimension matches the DataFrame 
            at self.data
        signal_name : str
            The label for this group of signals
        chan_names : list of str, optional
            The channel names for this data
        """
        logger.info(f'Adding continuous {signal_type} to the main DataFrame.')
        # Make MultiIndex columns
        midx = self._make_midx(signal_type, chan_names, cts_data.shape[1])
        # Build the DataFrame and attach it to the current dataframe
        new_data = pd.DataFrame(
            cts_data, index=self.data.index, columns=midx)
        self.data = pd.concat([self.data, new_data], axis=1)

    def add_trialized_data(self, trial_data, signal_type, chan_names=None):
        """Adds a trialized data field to the main DataFrame
        
        Parameters
        ----------
        trial_data : pd.DataFrame
            A trial_data dataframe containing a data field
            that will be added to the continuous dataframe
        signal_type : str
            The label for the data to be added
        chan_names : list of str, optional
            The channel names for the data when added
        """
        logger.info(f'Adding trialized {signal_type} to the main DataFrame')
        new_data = trial_data[['clock_time', signal_type]].set_index('clock_time')
        self.data = pd.concat([self.data, new_data], axis=1)
    
    def _make_midx(self, signal_type, chan_names=None, num_channels=None):
        """Creates a pd.MultiIndex for a given signal_type
        
        Parameters
        ----------
        signal_type : str
            Name of the signal type, to be used as the first level value
            of MultiIndex
        chan_names : list, optional
            Name of individual channels. If not provided,
            channel names will be automatically generated as
            ['0000', '0001', etc.]
        num_channels : int, optional
            Number of channels to create names for. Required if
            `chan_names` is not provided
        """
        if chan_names is None:
            if 'rates' in signal_type:
                # If merging rates, use the same names as the spikes
                chan_names = self.data.spikes.columns
            else:
                # Otherwise, generate names for the channels
                assert num_channels is not None, "`num_channels` must be provided if `chan_names` is not provided"
                chan_names = [f'{i:04d}' for i in range(num_channels)]
        # Create the MultiIndex for this data
        midx = pd.MultiIndex.from_product(
            [[signal_type], chan_names], names=('signal_type', 'channel'))
        return midx

    def calculate_onset(self, 
                        field_name, 
                        onset_threshold, 
                        peak_prominence=0.1,
                        peak_distance_s=0.1,
                        multipeak_threshold=0.2):
        """Calculates onset for a given field by finding 
        peaks and threshold crossings. Developed for 
        speed onset calculation

        Parameters
        ----------
        field_name : str
            The field to use for onset calculation, used
            with recursive getattr on self.data
        onset_threshold : float
            The threshold for onset as a percentage of the 
            peak height
        peak_prominence : float, optional
            Minimum prominence of peaks. Passed to 
            `scipy.signal.find_peaks`, by default 0.1
        peak_distance_s : float, optional
            Minimum distance between peaks. Passed to 
            `scipy.signal.find_peaks`, by default 0.1
        multipeak_threshold : float, optional
            Subsequent peaks within a trial must be no 
            larger than this percentage of the first peak, 
            otherwise the onset calculation fails, by default 0.2

        Returns
        -------
        pd.Series
            The times of identified peaks
        """

        import functools
        def rgetattr(obj, attr, *args):
            """A recursive drop-in replacement for getattr, 
            which also handles dotted attr strings
            """
            def _getattr(obj, attr):
                return getattr(obj, attr, *args)
            return functools.reduce(_getattr, [obj] + attr.split('.'))
        
        logger.info(f'Calculating {field_name} onset.')
        sig = rgetattr(self.data, field_name)
        # Find peaks
        peaks, properties = signal.find_peaks(
            sig,
            prominence=peak_prominence, 
            distance=peak_distance_s / (self.bin_width / 1000.0))
        peak_times = pd.Series(self.data.index[peaks])

        # Find the onset for each trial
        onset, onset_index = [], []
        for index, row in self.trial_info.iterrows():
            trial_start, trial_end = row['start_time'], row['end_time']
            # Find the peaks within the trial boundaries
            trial_peaks = peak_times[
                (trial_start < peak_times) & (peak_times < trial_end)]
            peak_signals = sig.loc[trial_peaks]
            # Remove trials with multiple larger peaks
            if multipeak_threshold is not None and len(trial_peaks) > 1:
                # Make sure the first peak is relatively large
                if peak_signals[0]*multipeak_threshold < peak_signals[1:].max():
                    continue
            elif len(trial_peaks) == 0:
                # If no peaks are found for this trial, skip it
                continue
            # Find the point just before speed crosses the threshold
            signal_threshold = onset_threshold * peak_signals[0]
            under_threshold = sig[trial_start:trial_peaks.iloc[0]] < signal_threshold
            if under_threshold.sum() > 0:
                onset.append(under_threshold[::-1].idxmax())
                onset_index.append(index)
        # Add the movement onset for each trial to the DataFrame
        onset_name = field_name.split('.')[-1] + '_onset'
        logger.info(f'`{onset_name}` field created in trial_info.')
        self.trial_info[onset_name] = pd.Series(onset, index=onset_index)
        
        return peak_times


''' Multiprocessing Functions '''
def smooth_column(args):
    """Low-level helper function for smoothing single column
    
    Parameters
    ----------
    args : tuple
        Tuple containing data to smooth in 1d array, 
        smoothing kernel in 1d array, whether to 
        ignore nans, and data dtype

    Returns
    -------
    np.ndarray
        1d array containing smoothed data
    """
    x, window, ignore_nans, dtype = args
    if ignore_nans and np.any(np.isnan(x)):
        x.astype(dtype)
        # split continuous data into NaN and not-NaN segments
        splits = np.where(np.diff(np.isnan(x)))[0] + 1
        seqs = np.split(x, splits)
        # if signal.convolve uses fftconvolve, there may be small negative values
        def rectify(arr):
            arr[arr < 0] = 0
            return arr
        # smooth only the not-NaN data
        seqs = [seq if np.any(np.isnan(seq)) else rectify(signal.convolve(seq, window, 'same')) for seq in seqs]
        # concatenate to single array
        y = np.concatenate(seqs)
    else:
        y = signal.convolve(x.astype(dtype), window, 'same')
    return y

def _poolmap(f, X, nprocs=multiprocessing.cpu_count(), chunksize=None):
    """Wrapper for multiprocessing.Pool.map() to parallelize
    execution of function f on elements of iterable X
    
    Parameters
    ----------
    f : function
        Function to execute in parallel
    X : iterable
        Iterable containing inputs to function f
    nprocs : int, optional
        Maximum number of parallel processes, by
        default the number of CPUs
    chunksize : int, optional
        Chunk size to fetch from iterable for 
        each process. Refer to multiprocessing 
        documentation for more information
    """
    with multiprocessing.Pool(processes=nprocs) as pool:
        out = pool.map(f, X, chunksize=chunksize)
    return out
