import mne
import logging
import numpy as np
from datamatrix._datamatrix._seriescolumn import _SeriesColumn
from datamatrix import series as srs, NAN, operations as ops

logger = logging.getLogger('eeg_eyetracking_parser')


class PupilEpochs(mne.Epochs):
    """An Epochs class for the PupilSize channel. This allows baseline
    correction to be applied to pupil size, even though this channel is not a
    regular data channel.
    """
    
    def __init__(self, *args, **kwargs):
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = True
        super().__init__(*args, **kwargs, picks='PupilSize')
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = False
        
    def average(self, *args, **kwargs):
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = True
        evoked = super().average(*args, **kwargs)
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = False
        return evoked


def epochs_to_series(dm, epochs, baseline_trim=(-2, 2)):
    """Takes an Epochs or PupilEpochs object and converts it to a DataMatrix
    SeriesColumn. If a baseline has been specified in the epoch, it is applied
    to each row of the series separately. Rows where the mean baseline value
    (z-scored) is not within the range indicated by `baseline_trim` are set to
    `NAN`.
    
    Parameters
    ----------
    dm: DataMatrix
        A DataMatrix object to which the series belongs
    epochs: Epochs or PupilEpochs
        The source object with the epoch data.
    baseline_trim: tuple of int, optional
        The range of acceptable baseline values. This refers to z-scores.
        
    Returns
    -------
    SeriesColumn
    """
    a = epochs.get_data()
    s = _SeriesColumn(dm, depth=a.shape[2])
    s._seq[epochs.metadata.index] = a.mean(axis=1)
    if epochs.baseline is not None:
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
