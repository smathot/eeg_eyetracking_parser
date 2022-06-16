"""A Python module for reading concurrently recorded EEG and eye-tracking data,
and parsing this data into convenient objects for further analysis.
"""

from ._triggers import trial_trigger, epoch_trigger
from ._parsing import read_subject
from ._pupil_epochs import PupilEpochs, epochs_to_series


__version__ = '0.1.1'
