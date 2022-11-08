import logging
from pathlib import Path
import numpy as np
import mne
from datamatrix import convert as cnv, io, functional as fnc
from datamatrix._datamatrix._seriescolumn import _SeriesColumn
from eyelinkparser import parse, defaulttraceprocessor
from ._triggers import trial_trigger, _parse_triggers, _validate_events
from . import _eeg_preprocessing as epp

logger = logging.getLogger('eeg_eyetracking_parser')


@fnc.memoize(persistent=True)
def read_subject(subject_nr, folder='data/', trigger_parser=None,
                 eeg_margin=30, min_sacc_dur=10, min_sacc_size=100,
                 min_blink_dur=10, blink_annotation='BLINK',
                 saccade_annotation='SACCADE', eeg_preprocessing=True,
                 save_preprocessing_output=True, plot_preprocessing=False,
                 eye_kwargs={}, downsample_data_kwargs={},
                 drop_unused_channels_kwargs={}, 
                 rereference_channels_kwargs={}, create_eog_channels_kwargs={},
                 set_montage_kwargs={}, annotate_emg_kwargs={},
                 band_pass_filter_kwargs={}, autodetect_bad_channels_kwargs={},
                 run_ica_kwargs={}, auto_select_ica_kwargs={},
                 interpolate_bads_kwargs={}):
    """Reads EEG, eye-tracking, and behavioral data for a single participant.
    This data should be organized according to the BIDS specification.
    
    EEG data is assumed to be in BrainVision data format (`.vhdr`, `.vmrk`,
    `.eeg`). Eye-tracking data is assumed to be in EyeLink data format (`.edf`
    or `.asc`). Behavioral data is assumed to be in `.csv` format.
    
    Metadata is taken from the behavioral `.csv` file if present, and from
    the eye-tracking data if not.
    
    __Important:__ This function uses persistent memoization, which means that
    the results for a given set of arguments are stored on disk and returned
    right away for subsequent calls. For more information, see
    <https://pydatamatrix.eu/memoization/>

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
    min_sacc_dur: int, optional
        The minimum duration of a saccade before it is annotated as a
        BAD_SACCADE.
    min_sacc_size: int, optional
        The minimum size of a saccade (in pixels) before it is annotated as a
        saccade.
    min_blink_dur: int, optional
        The minimum duration of a blink before it is annotated as a blink.
    blink_annotation: str, optional
        The annotation label to be used for blinks. Use a BAD_ suffix to
        use blinks a bads annotations.
    saccade_annotation: str, optional
        The annotation label to be used for saccades. Use a BAD_ suffix to
        use saccades a bads annotations.
    eeg_preprocessing: bool or list, optional
        Indicates whether EEG preprocessing should be performed. If `True`,
        then all preprocessing steps are performed. If a list is passed, then
        only those steps are performed for which the corresponding function
        name is in the list (e.g. `['downsample_data', 'set_montage']`)
    save_preprocessing_output: bool, optional
        Indicates whether output generated during EEG preprocessing should be
        saved.
    plot_preprocessing: bool, optional
        Indicates whether plots should be shown during EEG preprocessing.
    eye_kwargs: dict, optional
        Optional keyword arguments to be passed onto the EyeLink parser. If
        traceprocessor is provided, a default traceprocessor is used with
        advanced blink reconstruction enabled and 10x downsampling.
    downsample_data_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    drop_unused_channels_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    rereference_channels_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    create_eog_channels_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    set_montage_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    annotate_emg_kwargs: dict, optional
         Passed as keyword arguments to corresponding preprocessing function.
    band_pass_filter_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    autodetect_bad_channels_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    run_ica_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    auto_select_ica_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    interpolate_bads_kwargs: dict, optional
        Passed as keyword arguments to corresponding preprocessing function.
    
    Returns
    -------
    tuple:
        A raw (EEG data), events (EEG triggers), metadata (a table with
        experimental variables) tuple.
    """
    if isinstance(subject_nr, int):
        subject_path = Path(folder) / Path('sub-{:02d}'.format(subject_nr))
    else:
        subject_path = Path(folder) / Path('sub-{}'.format(subject_nr))
    logger.info(f'reading subject data from {subject_path}')
    raw, events = _read_eeg_data(subject_path / Path('eeg'), trigger_parser,
                                 eeg_margin)
    metadata = _read_beh_data(subject_path / Path('beh'))
    eye_path = subject_path / Path('eyetracking')
    dm = _read_eye_data(eye_path, metadata, eye_kwargs)
    if dm is not None and raw is not None:
        dm = _merge_eye_and_eeg_data(eye_path, raw, events, dm, min_sacc_dur,
                                     min_sacc_size, min_blink_dur,
                                     blink_annotation, saccade_annotation,
                                     eye_kwargs)
    if metadata is None and dm is not None:
        metadata = _dm_to_metadata(dm)
    if eeg_preprocessing:
        # Preprocessing output can be saved for visual inspection. By default
        # this is done in a subfolder of the participant data folder. This
        # only applies to certain preprocessing steps, i.e. those functions
        # that take a `preprocessing_path` keyword.
        if save_preprocessing_output:
            preprocessing_path = subject_path / Path('preprocessing')
            logger.info(f'saving preprocessing output to {preprocessing_path}')
            autodetect_bad_channels_kwargs['preprocessing_path'] = \
                preprocessing_path
            run_ica_kwargs['preprocessing_path'] = preprocessing_path
            auto_select_ica_kwargs['preprocessing_path'] = preprocessing_path
            annotate_emg_kwargs['preprocessing_path'] = preprocessing_path
        if plot_preprocessing:
            logger.info(f'creating plots during preprocessing')
            set_montage_kwargs['plot'] = True
            band_pass_filter_kwargs['plot'] = True
            autodetect_bad_channels_kwargs['plot'] = True
            auto_select_ica_kwargs['plot'] = True
            annotate_emg_kwargs['plot'] = True
        if isinstance(eeg_preprocessing, bool):
            eeg_preprocessing = [
                'drop_unused_channels',
                'rereference_channels',
                'annotate_emg',
                'downsample_data',
                'create_eog_channels',
                'set_montage',
                'band_pass_filter',
                'autodetect_bad_channels',
                'ica',
                'interpolate_bads']
        if 'drop_unused_channels' in eeg_preprocessing:
            logger.info('dropping unused channels')
            epp.drop_unused_channels(raw, **drop_unused_channels_kwargs)
        else:
            logger.info('*not* dropping unused channels')
        if 'rereference_channels' in eeg_preprocessing:
            logger.info('re-referencing channels')
            epp.rereference_channels(raw, **rereference_channels_kwargs)
        else:
            logger.info('*not* re-referencing channels')
        
        if 'annotate_emg' in eeg_preprocessing:
            logger.info('applying muscle artifact detection')
            epp.annotate_emg(raw, **annotate_emg_kwargs)
        else:
            logger.info('*not* applying muscle artifact detection')
        # Operations that require a high sampling rate, notable EMG annotation
        # should be done before downsampling. Everything else should be done
        # afterwards to save memory and time.
        if 'downsample_data' in eeg_preprocessing:
            logger.info('downsampling')
            raw, events = epp.downsample_data(raw, events,
                                              **downsample_data_kwargs)
        else:
            logger.info('*not* downsampling')
        if 'create_eog_channels' in eeg_preprocessing:
            logger.info('creating EOG channels')
            epp.create_eog_channels(raw, **create_eog_channels_kwargs)
        else:
            logger.info('*not* creating EOG channels')
        if 'set_montage' in eeg_preprocessing:
            logger.info('setting montage')
            epp.set_montage(raw, **set_montage_kwargs)
        else:
            logger.info('*not* setting montage')
        if 'band_pass_filter' in eeg_preprocessing:
            logger.info('applying band-pass filter')
            epp.band_pass_filter(raw, **band_pass_filter_kwargs)
        else:
            logger.info('*not* applying band-pass filter')
        if 'autodetect_bad_channels' in eeg_preprocessing:
            logger.info('autodetecting bad channels')
            epp.autodetect_bad_channels(raw, events,
                                        **autodetect_bad_channels_kwargs)
        else:
            logger.info('*not* autodetecting bad channels')
        if 'ica' in eeg_preprocessing:
            logger.info('running ICA')
            ica = epp.run_ica(raw, **run_ica_kwargs)
            logger.info('autoselect ICA')
            epp.auto_select_ica(raw, ica, **auto_select_ica_kwargs)
        else:
            logger.info('*not* running ICA')
        if 'interpolate_bads' in eeg_preprocessing:
            logger.info('interpolating bad channels')
            epp.interpolate_bads(raw, **interpolate_bads_kwargs)
        else:
            logger.info('*not* interpolating bad channels')
    return raw, events, metadata


def _dm_to_metadata(dm):
    """Convert dm to pandas dataframe but remove seriescolumns to save memory.
    """
    return cnv.to_pandas(dm[
        [col for name, col in dm.columns if not isinstance(col, _SeriesColumn)]
    ])


def _merge_eye_and_eeg_data(eye_path, raw, events,
                            dm, min_sacc_dur, min_sacc_size, min_blink_dur,
                            blink_annotation, saccade_annotation, eye_kwargs):
    """Add the eye data as channels to the EEG data. In addition, add blinks
    and saccades as annotations.
    """
    logger.info(f'merging eye-tracking and eeg data')
    # First read the eye-tracking data again, this time without downsampling
    # and without splitting the data into separate epochs. We do this, so that
    # we can merge this big dataset as channels into the EEG data.
    def only_trial(name): return name == 'trial'
    if 'traceprocessor' not in eye_kwargs:
        logger.info('no traceprocessor specified, using default')
        eye_kwargs['traceprocessor'] = defaulttraceprocessor(
            blinkreconstruct=True, mode='advanced')
    bigdm = parse(folder=eye_path, trialphase='trial', phasefilter=only_trial,
                  **eye_kwargs)
    # Valid rows are those rows in which there is an epoch 1. Since the epochs
    # are only parsed in dm (not in bigdm), we use indices in dm to filter both
    # dm and bigdm. The result is that dm and bigdm only contain rows that
    # should correspond to trials in the EEG recording.
    if '' in dm.t_onset_1:
        valid_rows = dm[dm[dm.t_onset_1] != '']
        logger.warning(
            f'ignoring {len(dm) - len(valid_rows)} rows of eye-tracking data without epoch 1')
        bigdm = bigdm[valid_rows]
        dm = dm[valid_rows]
    # Check for missing EEG triggers. If the first trigger is missing, a
    # ValueError is raised. If any other trigger is missing, it is marked and
    # removed also from dm and bigdm.
    triggers = trial_trigger(events)[:, 2]
    if triggers[0] != 128:
        raise ValueError(
            f'The first trial trigger is {triggers[0]}, should be 128')
    rows = list(range(len(dm)))
    for i, (tr1, tr2) in enumerate(zip(triggers[:-1], triggers[1:])):
        if tr2 - tr1 not in (1, -127):
            rows.remove(i + 1)
            logger.warning(f'missing trial trigger for trial {i + 1}')
    dm = dm[rows]
    bigdm = bigdm[rows]
    # Check if final triggers are missing from the EEG. If so, these are
    # truncated also from dm and bigdm. We also exclude the last trial,
    # because this last trial may not have been completed. (We don't check
    # this, but simply err on the side of caution.)
    missing = len(dm) - len(triggers)
    if missing > 0:
        logger.warning(
            f'final {missing} triggers missing from recording, truncating eye data with one extra trial because the last trial may be incomplete too')
        dm = dm[:-missing - 1]
        bigdm = bigdm[:-missing - 1]
        last_trigger_index = np.where(events[0][:, 2] >= 128)[0][-1]
        events = events[0][:last_trigger_index - 1], events[1]
    # Now double-check that EEG and eye-tracking data match in length
    n_trials_eeg = sum(events[0][:, 2] >= 128)
    assert(n_trials_eeg == len(dm))
    assert(n_trials_eeg == len(bigdm))
    logger.info(f'eeg data and metata have matching length')
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
        if len(raw) <= timestamp + trialdepth:
            logger.warning('eeg recording ended before last eye tracking trial')
            continue
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
    logger.info('adding blink and saccade annotations')
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
            if dur < min_blink_dur:
                continue
            start = st - t0
            onset.append((timestamp + start) / 1000)
            duration.append(dur / 1000)
            description.append(blink_annotation)
        # Saccades are deduced through the end of fixation n and the start of
        # fixation n + 1
        for st, sx, sy, et, ex, ey in zip(
                row.fixetlist_trial[:-1],
                row.fixxlist_trial[:-1],
                row.fixylist_trial[:-1],
                row.fixstlist_trial[1:],
                row.fixxlist_trial[1:],
                row.fixylist_trial[1:]):
            if np.isnan(et):
                break
            dur = et - st
            if dur < min_sacc_dur:
                continue
            start = st - t0
            size = ((sx - ex) ** 2 + (sy - ey) ** 2) ** .5
            if size < min_sacc_size:
                continue
            onset.append((timestamp + start) / 1000)
            duration.append(dur / 1000)
            description.append(saccade_annotation)
    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=description)
    raw.set_annotations(raw.annotations + annotations)
    return dm


def _read_eeg_data(eeg_path, trigger_parser, margin):
    """Reads eeg data and returns a raw, events tuple. If no eeg data is found,
    None, None is returned.
    """
    if not eeg_path.exists():
        logger.info('no eeg data detected')
        return None, None
    vhdr_path = list(eeg_path.glob('*.vhdr'))[0]
    logger.info(f'loading eeg data from {vhdr_path}')
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    if raw.info['sfreq'] != 1000:
        logger.warning(
            f'sampling rate is {raw.info["sfreq"]} Hz, resampling to 1000 Hz.')
        raw = raw.resample(1000)
    logger.info('creating events from annotations')
    events = mne.events_from_annotations(raw,
        _parse_triggers if trigger_parser is None else trigger_parser)
    if margin is not None:
        end = min(raw.times[-1], events[0][:, 0][-1] / 1000 + margin)
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
    logger.info('loading eye data from {eye_path}')
    return parse(folder=eye_path, maxtracelen=1, pupil_size=False,
                 gaze_pos=False)
