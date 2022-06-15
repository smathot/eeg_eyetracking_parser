import numpy as np

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
    triggers = events[0]
    return triggers[triggers[:, 2] >= 128]


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
