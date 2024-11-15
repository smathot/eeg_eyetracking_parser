import numpy as np
import logging

logger = logging.getLogger('eeg_eyetracking_parser')

TRIAL_TRIGGERS = list(range(128, 256))
TRIGGER_ANNOTATION_PREFIX = 'Stimulus/OVTK_StimulationId_Label_'


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
    events = _validate_events(events)
    if not 128 > trigger > 0:
        raise ValueError('trigger should be a number between 1 and 127')
    triggers = events
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
    events = _validate_events(events)
    triggers = events
    return triggers[triggers[:, 2] >= 128]


def _parse_triggers(label):
    """An internal function that converts labels as stored in our data."""
    if not label.startswith(TRIGGER_ANNOTATION_PREFIX):
        return None
    return 255 - int(label[-2:], 16)
    
    
def _validate_events(events):
    """Checks whether the events are in the correct format."""
    if isinstance(events, tuple):
        if len(events) != 2:
            raise ValueError(
                'events should be a numpy array or tuple of length 2')
        events = events[0]
    if not isinstance(events, np.ndarray):
        raise TypeError('events should be a numpy array')
    # Remove ghost triggers
    valid_events = []
    for i in range(len(events) - 1):
        dt = events[i + 1, 0] - events[i, 0]
        if dt == 1:
            logger.warning(
                f'ignoring ghost trigger: {events[i]} before {events[i + 1]}')
            continue
        valid_events.append(events[i])
    events = np.array(valid_events)
    # Keep only non-zero triggers
    events = events[events[:,2] != 0]
    codes = events[:, 2]
    if np.any((codes < 1) | (codes > 255)):
        raise ValueError('trigger codes should be values between 1 and 255')
    # Check for duplicate triggers within trials
    trialid = -1
    triggers_in_trial = []
    select_triggers = []
    for code in codes:
        # Detect trial triggers and always keep them
        if code >= 128:
            trialid += 1
            triggers_in_trial = []
            select_triggers.append(True)
            continue
        # Skip triggers that precede a trial onset
        if trialid == -1:
            logger.warning(f'trigger {code} precedes first trial')
            select_triggers.append(False)
            continue
        # Skip duplicate triggers within a trial
        if code in triggers_in_trial:
            logger.warning(
                f'duplicate trigger {code} in trial {trialid}, label {hex(255 - 128 - trialid % 128)}')
            select_triggers.append(False)
        else:
            select_triggers.append(True)
        triggers_in_trial.append(code)
    events = events[select_triggers]
    return events
