import mne
import numpy as np
from pathlib import Path
from datamatrix import convert as cnv, io
from eyelinkparser import parse, defaulttraceprocessor

TRIAL_TRIGGERS = list(range(128, 256))
TRIGGER_ANNOTATION_PREFIX = 'Stimulus/OVTK_StimulationId_Label_'
ZERO_TRIGGER_ANNOTATION = 'Stimulus/OVTK_StimulationId_Label_FF'


def epoch_trigger(events, trigger):
    """Selects a single epoch trigger from a tuple with event information.
    Epoch triggers have values between 1 and 127 (inclusive).

    Parameters
    ----------
    events: tuple
        Event information as returned by `read_subject()`.
    trigger: int
        A trigger code, which is a positive value.

    Returns
    -------
    array:
        A numpy array with events as expected by mne.Epochs().
    """
    _validate_events(events)
    if not 128 > trigger > 0:
        raise ValueError('trigger should be a number between 1 and 127')
    triggers = events[0]
    return triggers[triggers[:, 2] == trigger]
    

def trial_trigger(events):
    """Selects all trial triggers from event information. Trial triggers have
    values between 128 and 255 (inclusive).
    
    Parameters
    ----------
    events: tuple
        Event information as returned by `read_subject()`.

    Returns
    -------
    array:
        A numpy array with events as expected by mne.Epochs().
    """    
    _validate_events(events)
    return triggers[triggers[:, 2] >= 128]


def read_subject(subject_nr, folder='data/', trigger_parser=None,
                 eye_kwargs={}):
    """Reads EEG, eye-tracking, and behavioral data for a single participant.
    This data should be organized according to the BIDS specification.
    
    EEG data is assumed to be in BrainVision data format (`.vhdr`, `.vmrk`,
    `.eeg`). Eye-tracking data is assumed to be in EyeLink data format (`.edf`
    or `.asc`). Behavioral data is assumed to be in `.csv` format.
    
    Metadata is taken from the behavioral `.csv` file if present, and from
    the eye-tracking data if not.

    Parameters
    ----------
    subject_nr: int or sr
        The subject number to parse. If an int is passed, the subject number
        is assumed to be zero-padded to length two (e.g. '01'). If a string
        is passed, the string is used directly.
    folder: str, optional
        The folder in which the data is stored.
    trigger_parser: callable, optional
        A function that converts annotations to events. If no function is
        specified, triggers are assumed to be encoded by the OpenVibe
        acquisition software and to follow the convention for indicating
        trial numbers and event onsets as described in the readme.
    eye_kwargs: dict, optional
        Optional keyword arguments to be passed onto the EyeLink parser. If
        traceprocessor is provided, a default traceprocessor is used with
        advanced blink reconstruction enabled and 10x downsampling.

    Returns
    -------
    tuple:
        A raw (EEG data), events (EEG triggers), metadata (a table with
        experimental variables), eye_dm (eye-tracking data) tuple.
    """
    if isinstance(subject_nr, int):
        subject_path = Path(folder) / Path('sub-{:02d}'.format(subject_nr))
    else:
        subject_path = Path(folder) / Path('sub-{}'.format(subject_nr))
    raw, events = _read_eeg_data(subject_path / Path('eeg'), trigger_parser)
    metadata = _read_beh_data(subject_path / Path('beh'))
    dm = _read_eye_data(subject_path / Path('eyetracking'), metadata,
                        eye_kwargs)
    if metadata is None and dm is not None:
        metadata = cnv.to_pandas(dm)
    if events is not None and dm is not None:
        n_trials_eeg = sum(events[0][:, 2] >= 128)
        n_trials_eye = len(dm)
        assert(n_trials_eeg == n_trials_eye)
    return raw, events, metadata, dm


def _read_eeg_data(eeg_path, trigger_parser):
    """Reads eeg data and returns a raw, events tuple. If no eeg data is found,
    None, None is returned.
    """
    if not eeg_path.exists():
        return None, None
    vhdr_path = list(eeg_path.glob('*.vhdr'))[0]
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    events = mne.events_from_annotations(raw,
        _parse_triggers if trigger_parser is None else trigger_parser)
    _validate_events(events)
    return raw, events


def _read_beh_data(beh_path):
    """Reads behavioral data and returns it as a DataFrame. None is returned
    if no data is found.
    """
    if not beh_path.exists():
        return None
    csv_path = list(beh_path.glob('*.csv'))[0]
    return cnv.to_pandas(io.readtxt(csv_path))


def _read_eye_data(eye_path, metadata, kwargs):
    """Reads eye-tracking data and returns it as a DataMatrix. None is returned
    if no data is found.
    """
    if not eye_path.exists():
        return None
    if 'traceprocessor' not in kwargs:
        kwargs['traceprocessor'] = defaulttraceprocessor(
            blinkreconstruct=True, downsample=10, mode='advanced')
    return parse(folder=eye_path, **kwargs)


def _parse_triggers(label):
    """An internal function that converts labels as stored in our data."""
    if not label.startswith(TRIGGER_ANNOTATION_PREFIX):
        return None
    if label == ZERO_TRIGGER_ANNOTATION:
        return None
    return 255 - int(label[-2:], 16)
    
    
def _validate_events(events):
    """Checks whether the events are in the correct format."""
    if not isinstance(events, tuple):
        raise TypeError('events should be a tuple')
    if len(events) != 2:
        raise ValueError('events should be a tuple of length 2')
    if not isinstance(events[0], np.ndarray):
        raise TypeError('the first element of events should be a numpy array')
    if not isinstance(events[1], dict):
        raise TypeError('the second element of events should be a dict')
    codes = events[0][:, 2]
    if np.any((codes < 1) | (codes > 255)):
        raise ValueError('trigger codes should be values between 1 and 255')
