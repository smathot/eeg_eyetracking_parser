import mne
import logging
import warnings
import numpy as np
from datamatrix._datamatrix._seriescolumn import _SeriesColumn
from datamatrix import series as srs, NAN, operations as ops, functional as fnc
from functools import lru_cache
import autoreject
import os

logger = logging.getLogger('eeg_eyetracking_parser')


class PupilEpochs(mne.Epochs):
    """An Epochs class for the PupilSize channel. This allows baseline
    correction to be applied to pupil size, even though this channel is not a
    regular data channel. In addition, this class allows pupil sizes to be
    excluded based on deviant baseline values, which is recommended for
    pupil analysis (but not typically done for eeg).
    
    Parameters
    ----------
    *args: iterable
        Arguments passed to mne.Epochs()
    baseline_trim: tuple of int, optional
        The range of acceptable baseline values. This refers to z-scores.
    **kwargs: dict
        Keywords passed to mne.Epochs()

    Returns
    -------
    Epochs:
        An mne.Epochs() object with autorejection applied.
    
    """
    def __init__(self, *args, baseline_trim=(-2, 2), **kwargs):
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = True
        if 'preload' in kwargs and not kwargs['preload']:
            raise ValueError('PupilEpochs must be preloaded')
        kwargs['preload'] = True
        super().__init__(*args, **kwargs, picks='PupilSize')
        if baseline_trim is not None and self.baseline is not None:
            self._baseline_trim(baseline_trim)
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = False
        
    def average(self, *args, **kwargs):
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = True
        evoked = super().average(*args, **kwargs)
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = False
        return evoked
    
    def _baseline_trim(self, trim):
        # Get the indices that correspond to baseline samples
        start, end = self.baseline
        t = self._raw_times
        i = np.where((t >= start) & (t <= end))[0]
        # An array of z-scored baseline pupil sizes per trial
        baseline_sizes = np.nanmean(self._data[:, 0, i], axis=1)
        baseline_sizes -= np.nanmean(baseline_sizes)
        baseline_sizes /= np.nanstd(baseline_sizes)
        # Indices of outliers
        out_i = np.where((baseline_sizes < trim[0])
                         | (baseline_sizes > trim[1]))
        # Set all outliers to nan
        self._data[out_i] = np.nan
        logger.warning(
            f'excluding {len(out_i[0])} trials based on baseline pupil size')


@fnc.memoize(persistent=True)
def autoreject_epochs(*args, ar_kwargs=None, **kwargs):
    """A factory function that creates an Epochs() object, applies
    autorejection, and then returns it.
    
    __Important:__ This function uses persistent memoization, which means that
    the results for a given set of arguments are stored on disk and returned
    right away for subsequent calls. For more information, see
    <https://pydatamatrix.eu/memoization/>

    Parameters
    ----------
    *args: iterable
        Arguments passed to mne.Epochs()
    ar_kwargs: dict or None, optional
        Keywords to be passed to AutoReject(). If `n_interpolate` is not
        specified, a default value of [1, 4, 8, 16] is used.
    **kwargs: dict
        Keywords passed to mne.Epochs()

    Returns
    -------
    Epochs:
        An mne.Epochs() object with autorejection applied.
    """
    # autoreject is initially run on all EEG channels, and only afterwards are
    # channels picked (if any picks are specified)
    if 'picks' in kwargs:
        picks = kwargs['picks']
        del kwargs['picks']
    else:
        picks is None
    epochs = mne.Epochs(*args, preload=True, picks='eeg', **kwargs)
    if ar_kwargs is None:
        ar_kwargs = {}
    if 'n_interpolate' not in ar_kwargs:
        ar_kwargs['n_interpolate'] = [1, 4, 8, 16]
    ar = autoreject.AutoReject(**ar_kwargs)
    epochs = ar.fit_transform(epochs)
    if picks is not None:
        epochs.pick_channels(picks)
    return epochs


def epochs_to_series(dm, epochs, baseline_trim=None):
    """Takes an Epochs, PupilEpochs, or EpochsTFR object and converts it to a
    DataMatrix SeriesColumn. If a baseline has been specified in the epoch, it
    is applied to each row of the series separately. Rows where the mean
    baseline value (z-scored) is not within the range indicated by
    `baseline_trim` are set to `NAN`.

    Parameters
    ----------
    dm: DataMatrix
        A DataMatrix object to which the series belongs
    epochs: Epochs, PupilEpochs, or EpochsTFR
        The source object with the epoch data.
    baseline_trim: tuple of int, optional
        The range of acceptable baseline values. This refers to z-scores.

    Returns
    -------
    SeriesColumn
    """
    warnings.warn('use datamatrix.convert.from_mne_epochs() instead',
                  DeprecationWarning)
    # Different objects have different ways to access the data
    if hasattr(epochs, 'get_data') and callable(epochs.get_data):
        a = epochs.get_data()
    elif hasattr(epochs, 'data'):
        a = epochs.data
    else:
        raise TypeError('invalid epochs object')
    # If the data is four dimensional, then it's probably a time-frequency
    # object where the second dimension reflects the frequency bands while
    # the third dimension reflects channels. We average over both.
    if len(a.shape) == 4:
        a = a.mean(axis=(1, 2))
    # If the data is three dimension, then we only average over channels.
    else:
        a = a.mean(axis=1)
    s = _SeriesColumn(dm, depth=a.shape[1])
    s._seq[epochs.metadata.index] = a
    if hasattr(epochs, 'baseline') and epochs.baseline is not None:
        start, end = epochs.baseline
        t = epochs._raw_times
        i = np.where((t >= start) & (t <= end))[0]
        bl_start = i[0]
        bl_end = i[-1]
        if baseline_trim is not None:
            bl = srs.reduce(s[:, bl_start : bl_end])
            for row_nr, (z, bl) in enumerate(zip(ops.z(bl), bl)):
                if z < baseline_trim[0] or z > baseline_trim[1]:
                    logging.debug(
                        f'setting trace to NAN because baseline out of bounds (bl={bl}, z={z})')
                    s[row_nr] = NAN
        s = srs.baseline(s, s, bl_start, bl_end)
    return s


def tfr_to_surface(dm, epochs):
    """Takes an EpochsTFR object and converts it to a DataMatrix
    MultiDimensionalColumn of two dimensions.

    Parameters
    ----------
    dm: DataMatrix
        A DataMatrix object to which the series belongs
    epochs: EpochsTFR
        The source object with the epoch data.

    Returns
    -------
    SeriesColumn
    """
    warnings.warn('use datamatrix.convert.from_mne_tfr() instead',
                  DeprecationWarning)
    try:
        from datamatrix._datamatrix._multidimensionalcolumn import \
            _MultiDimensionalColumn
    except ImportError:
        raise Exception(
            'Your version of DataMatrix does not support multidimensional columns')
    # Different objects have different ways to access the data
    try:
        a = epochs.data
    except AttributeError:
        raise TypeError('invalid EpochsTFR object')
    if len(a.shape) != 4:
        raise TypeError('invalid epochs object')
    a = a.mean(axis=1)
    s = _MultiDimensionalColumn(dm, shape=a.shape[1:])
    s._seq[epochs.metadata.index] = a
    return s
