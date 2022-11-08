"""A Python module for reading concurrently recorded EEG and eye-tracking data,
and parsing this data into convenient objects for further analysis.
"""
import logging
from ._triggers import trial_trigger, epoch_trigger
from ._parsing import read_subject
from ._custom_epochs import PupilEpochs, autoreject_epochs, epochs_to_series, \
    tfr_to_surface


__version__ = '0.10.0'
logger = logging.getLogger('eeg_eyetracking_parser')
logger.info(f'eeg_eyetracking_parser {__version__}')
