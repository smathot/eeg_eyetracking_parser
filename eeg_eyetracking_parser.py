import logging
from pathlib import Path
import numpy as np
import mne
from datamatrix import convert as cnv, io
from eyelinkparser import parse, defaulttraceprocessor

TRIAL_TRIGGERS = list(range(128, 256))
TRIGGER_ANNOTATION_PREFIX = 'Stimulus/OVTK_StimulationId_Label_'
ZERO_TRIGGER_ANNOTATION = 'Stimulus/OVTK_StimulationId_Label_FF'
logger = logging.getLogger('eeg_eyetracking_parser')


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


def read_subject(subject_nr, folder='data/', trigger_parser=None,
                 eeg_margin=30, eye_kwargs={}):
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
    eeg_margin: int, optional
        The number of seconds after the last trigger to keep. The rest of the
        data will be cropped to save memory (in case long periods of extraneous
        data were recorded).
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
    logger.info(f'reading subject data from {subject_path}')
    raw, events = _read_eeg_data(subject_path / Path('eeg'), trigger_parser,
                                 eeg_margin)
    # _trim_eeg_data(raw, events, eeg_margin)
    metadata = _read_beh_data(subject_path / Path('beh'))
    eye_path = subject_path / Path('eyetracking')
    dm = _read_eye_data(eye_path, metadata, eye_kwargs)
    if dm is not None and raw is not None:
        _merge_eye_and_eeg_data(eye_path, raw, events, dm)
    if metadata is None and dm is not None:
        metadata = cnv.to_pandas(dm)
    if events is not None and dm is not None:
        n_trials_eeg = sum(events[0][:, 2] >= 128)
        n_trials_eye = len(dm)
        assert(n_trials_eeg == n_trials_eye)
        logger.info(f'eeg data and metata have matching length')
    return raw, events, metadata, dm


def _merge_eye_and_eeg_data(eye_path, raw, events, dm):
    """Add the eye data as channels to the EEG data. In addition, add blinks
    and saccades as annotations.
    """
    logger.info(f'merging eye-tracking and eeg data')
    # First read the eye-tracking data again, this time without downsampling
    # and without splitting the data into separate epochs. We do this, so that
    # we can merge this big dataset as channels into the EEG data.
    def only_trial(name): return name == 'trial'
    bigdm = parse(folder=eye_path, trialphase='trial', phasefilter=only_trial)
    # The start of the trial as recorded by the EEG start-trial marker and the
    # eyelink start_trial message are usually a little offset. However, the
    # EEG epoch triggers should *not* be offset relative to the eye-tracking
    # start_phase messages. (If they are, the experiment is not correctly
    # programmed). Therefore, we use the first epoch trigger to determine the
    # offset of the eye-movement data relative to the EEG data. We add this to
    # both the eye-tracking dms as the `eye_offset` column.
    for i, (row, bigrow) in enumerate(zip(dm, bigdm)):
        eye_t0 = bigrow.ttrace_trial[0]
        eye_t1 = row.t_onset_1 - eye_t0
        triggers = events[0]
        trigger_index = np.where(triggers[:, 2] >= 128)[0][i]
        eeg_t0, eeg_t1 = triggers[trigger_index:trigger_index + 2][:, 0]
        eeg_t1 -= eeg_t0
        eye_offset = eeg_t1 - eye_t1
        bigrow.eye_offset = eye_offset
        dm.eye_offset = eye_offset
    logger.info(f'trial offset (eye - eeg): {dm.eye_offset.mean}')
    # Now add gaze coordinates and pupil size as channels to the raw object.
    # Periods of missing data (in between recordings) is set to the median.
    logger.info('adding GazeX, GazeY, and PupilSize channels')
    data = np.empty((3, len(raw)), dtype=float)
    data[:] = np.nan
    trialdepth = bigdm.ptrace_trial.depth
    for (timestamp, _, code), row in zip(trial_trigger(events), bigdm):
        timestamp += row.eye_offset
        data[0, timestamp: timestamp + trialdepth] = row.xtrace_trial
        data[1, timestamp: timestamp + trialdepth] = row.ytrace_trial
        data[2, timestamp: timestamp + trialdepth] = row.ptrace_trial
    data[0, np.isnan(data[0])] = np.nanmedian(data[0])
    data[1, np.isnan(data[1])] = np.nanmedian(data[1])
    data[2, np.isnan(data[2])] = np.nanmedian(data[2])
    ch_names = ['GazeX', 'GazeY', 'PupilSize']
    ch_types = 'misc'
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=raw.info['sfreq'],
        ch_types=ch_types)
    tmp = mne.io.RawArray(data, info)
    raw.add_channels([tmp])
    # Finally, get the blinks and saccades from the eye-tracking data and add
    # them as BAD annotations to the raw.
    logger.info('adding BAD_BLINK and BAD_SACCADE annotations')
    onset = []
    duration = []
    description = []
    for (timestamp, _, code), row in zip(trial_trigger(events), bigdm):
        timestamp += row.eye_offset
        t0 = row.ttrace_trial[0]
        for st, et in zip(row.blinkstlist_trial, row.blinketlist_trial):
            if np.isnan(st):
                break
            dur = et - st
            start = st - t0
            onset.append((timestamp + start) / 1000)
            duration.append(dur / 1000)
            description.append('BAD_BLINK')
        # Saccades are deduced through the end of fixation n and the start of
        # fixation n + 1
        for st, et in zip(row.fixetlist_trial[:-1], row.fixstlist_trial[1:]):
            if np.isnan(et):
                break
            dur = et - st
            start = st - t0
            onset.append((timestamp + start) / 1000)
            duration.append(dur / 1000)
            description.append('BAD_SACCADE')
    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=description)
    raw.set_annotations(raw.annotations + annotations)


def _read_eeg_data(eeg_path, trigger_parser, margin):
    """Reads eeg data and returns a raw, events tuple. If no eeg data is found,
    None, None is returned.
    """
    if not eeg_path.exists():
        logger.info('no eeg data detected')
        return None, None
    vhdr_path = list(eeg_path.glob('*.vhdr'))[0]
    logger.info('loading eeg data from {vhdr_path}')
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    logger.info('creating events from annotations')
    events = mne.events_from_annotations(raw,
        _parse_triggers if trigger_parser is None else trigger_parser)
    if margin is not None:
        end = min(len(raw), events[0][:, 0][-1] / 1000 + margin)
        logger.info(f'trimming eeg to 0 - {end} s')
        raw.crop(0, end)
    logger.info('validating events')
    _validate_events(events)
    logger.info('creating annotations from events')
    raw.set_annotations(
        mne.annotations_from_events(
            events[0],
            sfreq=raw.info['sfreq']))
    return raw, events


def _read_beh_data(beh_path):
    """Reads behavioral data and returns it as a DataFrame. None is returned
    if no data is found.
    """
    if not beh_path.exists():
        logger.info('no behavioral data detected')
        return None
    csv_path = list(beh_path.glob('*.csv'))[0]
    logger.info('loading behavioral data from {csv_path}')
    return cnv.to_pandas(io.readtxt(csv_path))


def _read_eye_data(eye_path, metadata, kwargs):
    """Reads eye-tracking data and returns it as a DataMatrix. None is returned
    if no data is found.
    """
    if not eye_path.exists():
        logger.info('no eye data detected')
        return None
    if 'traceprocessor' not in kwargs:
        logger.info('no traceprocessor specified, using default')
        kwargs['traceprocessor'] = defaulttraceprocessor(
            blinkreconstruct=True, downsample=10, mode='advanced')
    logger.info('loading eye data from {eye_path}')
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
