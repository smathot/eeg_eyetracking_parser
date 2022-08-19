import numpy as np
import mne
import os
from ._triggers import trial_trigger
from mne.preprocessing import ICA, annotate_muscle_zscore
import autoreject as ar
import json
import matplotlib.pyplot as plt


def rereference_channels(raw, ref_channels=['A1', 'A2']):
    """
    Re-reference EEG data to the mastoids

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    ref_channels: string, list of strings, optional
        The channels that you want to re-reference to

    Returns
    -------
    raw: instance of raw EEG data
        A raw re-referenced EEG data
    """
    raw.set_eeg_reference(ref_channels=ref_channels)
    raw.drop_channels(ref_channels)


def drop_unused_channels(raw, chan_name_pattern='Channel'):
    """
    Drop unused channels. The pattern is that they all start with "Channel"

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    chan_name_pattern: string, optional
        All the channels that start with chan_name pattern will be excluded as
        they contain no data

    Returns
    -------
    raw: instance of raw EEG data
        A raw EEG data with empty channels removed
    """
    empty_chan_names = [chan_name for chan_name in raw.info['ch_names']\
                        if chan_name.startswith(chan_name_pattern)]
    raw.drop_channels(empty_chan_names)


def create_eog_channels(raw,
                        eog_channels=['VEOGT', 'VEOGB', 'HEOGL', 'HEOGR']):
    """
    Create EOG channels by subtracting corresponding channels
    (VEOG = VEOGT - VEOGB)
    (HEOG = HEOGL - HEOGR)

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    eog_channels: list of strings, optional
        VEOG top, VEOG bottom, HEOG left and HEOG right

    Returns
    -------
    raw: instance of raw EEG data
        A raw EEG data with two EOG channels
    """
    chinds = [raw.ch_names.index(ichannel) for ichannel in eog_channels]
    veog = raw._data[chinds[0], :] - raw._data[chinds[1], :]
    heog = raw._data[chinds[2], :] - raw._data[chinds[3], :]
    veog = veog[np.newaxis, :]
    heog = heog[np.newaxis, :]
    info = mne.create_info(['VEOG', 'HEOG'], raw.info['sfreq'], ['eog', 'eog'])
    neweog = mne.io.RawArray(np.r_[veog, heog], info)
    raw.add_channels([neweog], force_update_info=True)
    raw.set_channel_types(dict(HEOG='eog', VEOG='eog'))
    raw.drop_channels(eog_channels)


def set_montage(raw, montage_name='standard_1020', plot=False):
    """
    Set Montage for EEG channel positions
    Here we use standard_1020 and thus electrode names and positions are
    according to international 10-20 system

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    montage_name: string, optional
        A montage that contains channel positions
    plot: bool, optional
        True if plotting is to be done. By default, no plotting

    Returns
    -------
    raw: instance of raw EEG data
        A raw EEG data with channel locations
    """
    montage = mne.channels.make_standard_montage(montage_name)
    if plot:
        montage.plot()
    if 'Digi' in raw.info['ch_names']:
        raw.set_channel_types({'Digi': 'stim'})
    raw.set_montage(montage, match_case=False)


def annotate_emg(raw, threshold=5, ch_type="eeg",
                 muscle_frequencies=[110, 140], plot=False,
                 preprocessing_path=None, subject_nr=None):
    """
    Annotate segments of data that represent muscle artifacts

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    threshold: float, optional
        The threshold in z-scores for marking data as muscle artifacts
    ch_type: string, optional
        Type of channels we select, here - eeg
    muscle_frequencies: list, optional
        Frequencies of bandpass filter, as muscle artifacts are noticable
        primarily at high frequencies
    plot: bool, optional
        True if plotting is to be done. By default, no plotting
    preprocessing_path: string, optional
        If the output is saved, specify the preprocessing path it will be saved
        to
    subject_nr: int, optional
        Specify subject_nr for saving to correct folder

    Returns
    -------
    raw: instance of raw EEG data
        A raw EEG data with segments of data annototed with regard to muscle
        artifacts
    """
    muscle_annotations, muscle_scores = annotate_muscle_zscore(
        raw, ch_type=ch_type, threshold=threshold,
        filter_freq=muscle_frequencies)
    raw.set_annotations(raw.annotations + muscle_annotations)
    if plot and preprocessing_path is not None:
        emg_dir = os.path.join(preprocessing_path, "EMG")
        subject_emg_dir = os.path.join(emg_dir, 'subject_' + str(subject_nr))
        if not os.path.exists(subject_emg_dir):
            os.makedirs(subject_emg_dir)
        fig, ax = plt.subplots()
        ax.plot(raw.times, muscle_scores)
        ax.axhline(y=threshold, color='blue')
        ax.set(xlabel='time in s', ylabel='zscore')
        plt.savefig(
            os.path.join(subject_emg_dir, 'muscle_annotations_zscore.png'))
        plt.show()


def notch_filter(raw, frequencies_remove=(50, 100, 150, 200)):
    """
    Removes power line noise generated by EEG system

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    frequencies_remove: float, array of float, optional
        Frequencies to remove from the data

    Returns
    -------
    raw: instance of raw EEG data
        A raw EEG data with certain frequencies that were filtered out
    """
    raw.notch_filter(frequencies_remove, picks=['eeg', 'eog'])


def band_pass_filter(raw, lf=0.1, hf=40, plot=False):
    """
    High-pass and low-pass filter for EEG data (band-pass filter)
    High-pass removes slow drifts
    Low-pass removes the short-term fluctuations

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    lf: float, optional
        Low-pass frequency edge
    hf: float, optional
        High-pass frequency edge
    plot: bool, optional
        True if plotting is to be done. By default, no plotting

    Returns
    -------
    raw: instance of raw EEG data
        A raw filtered EEG data
    """
    if plot:
        raw_unfiltered = raw.copy()
    raw.filter(l_freq=lf, h_freq=hf, picks=['eeg', 'eog'])
    if plot:
        for title, data in zip(['Unfiltered', 'Bandpass filtered'],
                               [raw_unfiltered, raw]):
            fig = data.plot_psd(fmax=min(250, raw.info['sfreq'] / 2))
            fig.suptitle(title)


def downsample_data(raw, events, srate=250):
    """
    Downsample EEG data for computational efficacy

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    events: tuple
        Event information as returned by `read_subject()`.
    srate: float, optional
        Sampling rate to convert to

    Returns
    -------
    raw: instance of raw EEG data
        A raw downsampled EEG data
    events: tuple
        Downsampled events.
    """
    events, event_id = events
    raw, events = raw.resample(srate, events=events, npad="auto")
    return raw, (events, event_id)


def autodetect_bad_channels(raw, events, plot=False, preprocessing_path=None, 
                            subject_nr=None, eeg_scaling=20e-5):
    """
    Detect bad channels using ransac algorithm

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    events: tuple
        Event information as returned by `read_subject()`.
    plot: bool, optional
        True if plotting is to be done. By default, no plotting
    preprocessing_path: string, optional
        If the output is saved, specify the preprocessing path it will be saved
        to
    subject_nr: int, optional
        Specify subject_nr for saving to correct folder
    eeg_scaling: dict, optional
        Scaling with which to visualize the data, in µV

    Returns
    -------
    raw: instance of raw EEG data
        A raw EEG data with bad channels detected (raw.info['bads])
    """
    raw.info['bads'] = []
    raw_bads = raw.copy()
    raw_bads.pick_types(eeg=True)
    trial_events = trial_trigger(events)
    median_duration = np.median(
        trial_events[1:, 0] - trial_events[:-1, 0]) * 1.5
    cue_epoch = mne.Epochs(raw_bads, trial_events, tmin=0, baseline=None,
                           tmax=median_duration / 1000, preload=True)
    ransac = ar.Ransac()
    ransac = ransac.fit(cue_epoch)
    raw.info['bads'] = ransac.bad_chs_
    if plot or preprocessing_path is not None:
        bad_dir = os.path.join(preprocessing_path, "Bad_channels")
        subject_bad_dir = os.path.join(
            bad_dir, 'subject_' + str(subject_nr))
        if not os.path.exists(subject_bad_dir):
            os.makedirs(subject_bad_dir)
        if plot:
            raw_bads.plot(butterfly=True, bad_color='r', scalings=eeg_scaling)
            plt.savefig(
                os.path.join(subject_bad_dir, 'raw_with_bads_ransac.png'))
            plt.show()
            plt.figure(figsize=(20, 5))
            plt.bar(range(len(raw_bads.ch_names)), ransac.bad_log.mean(0))
            plt.xticks(range(len(raw_bads.ch_names)), raw_bads.ch_names)
            plt.tight_layout()
            plt.savefig(os.path.join(subject_bad_dir, 'ransac_scores.png'))
            plt.show()
        if preprocessing_path is not None:
            bads_txt_file = os.path.join(subject_bad_dir, 'bads.txt')
            raw_bads_file = os.path.join(subject_bad_dir, 'raw_bads_eeg.fif')
            with open(bads_txt_file, 'w') as f:
                json.dump(raw.info['bads'], f)
            raw.save(raw_bads_file, overwrite=True)


def run_ica(raw, lf=1, sel_components='all', ica_method='picard', n_iter=500,
            random_state_set=97, preprocessing_path=None, subject_nr=None):
    """
    Run ICA (independent component analysis)
    Here we use a robust and fast 'picard' method by default as a fast and
    robust algorithm

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    lf: float, optional
        Low-pass frequency edge
    sel_components: string, int, float, optional
        Be default, 'all' meaning that number of components = number of
        channels
    ica_method: string, optional
        Choice of the following methods: ‘fastica’, ‘infomax’ or ‘picard’
    n_iter: int, optional
        Maximum number of iterations
    random_state_set:int, optional
        Seed for random generator
    preprocessing_path: string, optional
        If the output is saved, specify the preprocessing path it will be saved
        to
    subject_nr: int, optional
        Specify subject_nr for saving to correct folder

    Returns
    -------
    ica: instance of ICA
        A result of independent component analysis
    """
    raw_ica = raw.copy().filter(l_freq=lf, h_freq=None)

    if sel_components == 'all':
        picks = mne.pick_types(raw.info, eeg=True)
        n_sel_components = len(picks)
    else:
        n_sel_components = sel_components
    ica = ICA(n_components=n_sel_components, method=ica_method,
              max_iter=n_iter, random_state=random_state_set)
    ica.fit(raw_ica)
    if preprocessing_path is not None:
        ica_dir = os.path.join(preprocessing_path, "ICA")
        subject_ica_dir = os.path.join(
            ica_dir, 'subject_' + str(subject_nr))
        if not os.path.exists(subject_ica_dir):
            os.makedirs(subject_ica_dir)
        ica.save(os.path.join(subject_ica_dir, 'res_ica.fif'), overwrite=True)
    return ica


def auto_select_ica(raw, ica, plot=False, preprocessing_path=None,
                    subject_nr=None):
    """
    Select ICA components automatically by matching them to EOG channels

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    ica: instance of ICA
        A result of independent component analysis
    plot: bool, optional
        True if plotting is to be done. By default, no plotting
    preprocessing_path: string, optional
        If the output is saved, specify the preprocessing path it will be saved
        to
    subject_nr: int, optional
        Specify subject_nr for saving to correct folder

    Returns
    -------
    raw: instance of raw EEG data
        A raw EEG data with ICA components removed
    """
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    ica.apply(raw)
    if preprocessing_path is None:
        return
    ica_dir = os.path.join(preprocessing_path, "ICA")
    subject_ica_dir = os.path.join(
        ica_dir, 'subject_' + str(subject_nr))
    if not os.path.exists(subject_ica_dir):
        os.makedirs(subject_ica_dir)
    if plot:
        for ieog in eog_indices:
            ica.plot_properties(raw, picks=ieog)
            plt.savefig(os.path.join(
                subject_ica_dir, str(ieog) + 'ica_component_properties.png'))
            plt.show()
        ica.plot_sources(raw, show_scrollbars=False)
        plt.savefig(os.path.join(subject_ica_dir, 'latent_sources_raw.png'))
        plt.show()
        ica.plot_scores(eog_scores)
        plt.savefig(os.path.join(subject_ica_dir, 'ica_scores.png'))
        plt.show()
    ica_txt_file = os.path.join(subject_ica_dir, 'ica_removed.txt')
    ica_components = ['ICA component ' + str(x) for x in eog_indices]
    with open(ica_txt_file, 'w') as f:
        json.dump(ica_components, f)


def interpolate_bads(raw, resetting=False):
    """
    Interpolate channels that we previously marked as bads

    Parameters
    ----------
    raw: instance of raw EEG data
        A raw EEG data
    resetting: bool, optional
        If False bads are still mentioned in 'info'

    Returns
    -------
    raw: instance of raw EEG data
        A raw EEG data with bad channels interpolated
    """
    raw.interpolate_bads(reset_bads=resetting)
