import numpy as np
from mne.time_frequency import tfr_morlet, tfr_multitaper


def run_morlet(epochs, frequency_start, frequency_end, frequency_step=1, pick_channels=None):
    """
    Run time-frequency decomposition using Morlet wavelets

    Parameters
    ----------
    epochs: Epochs
        The source object with the epoch data.
    frequency_start: int
        The beginning of frequencies you are interested in (in Hz)
    frequency_end: int
        The end of frequencies you are interested in (in Hz)
    frequency_step: int, optional
        Step with which frequencies will be analyzed. By default it is set to 1,
        meaning that each frequency from beginning to end will be taken (in Hz)
    pick_channels: str, list of strings, optional
        The channels you want to compute time-frequency for. By default, it is set to None,
        meaning that all channels will be used

    Returns
    -------
    average_power: average TFR
        Average power from the specified frequencies and channels
    """
    frequencies = np.arange(frequency_start, frequency_end, frequency_step)
    n_cycles = frequencies / 2.
    if pick_channels:
        sel_channels = [epochs.ch_names.index(pick_channels)]
    else:
        sel_channels = None

    average_power = tfr_morlet(epochs, freqs=frequencies, return_itc=False, n_cycles=n_cycles, picks=sel_channels)
    return average_power

def run_multitaper(epochs, frequency_start, frequency_end, time_bandwidth, frequency_step=1):
    """
    Run time-frequency decomposition using DPSS tapers

    Parameters
    ----------
    epochs: Epochs
        The source object with the epoch data.
    frequency_start: int
        The beginning of frequencies you are interested in (in Hz)
    frequency_end: int
        The end of frequencies you are interested in (in Hz)
    time_bandwidth: float
        Number of tapers (time_bandwidth-1). If None, will be set to 4 (3 tapers).
    frequency_step: int, optional
        Step with which frequencies will be analyzed. By default it is set to 1,
        meaning that each frequency from beginning to end will be taken (in Hz)

    Returns
    -------
    average_power: average TFR
        Average power from the specified frequencies and channels
    """
    frequencies = np.arange(frequency_start, frequency_end, frequency_step)
    n_cycles = frequencies / 2.

    average_power = tfr_multitaper(epochs, freqs=frequencies, return_itc=False,
                                   time_bandwidth=time_bandwidth, n_cycles=n_cycles)
    return average_power
