# Python parser for combined EEG and eye-tracking data 

Copyright (2022) Hermine Berberyan, Wouter Kruijne, Sebastiaan Mathôt, Ana Vilotijević

## About

A Python module for reading concurrently recorded EEG and eye-tracking data, and parsing this data into convenient objects for further analysis. For this to work, several assumptions need to be met, as described under [Assumptions](#assumptions). At present, this module is largely for internal use, and focused on our own recording environment.

Key features:

- Experimental variables (such as conditions) from the eye-tracking data are used as metadata for the EEG analysis.
- Gaze and pupil data is added as channels to the EEG data.
- Blinks and saccades from the eye-tracking data are added as BAD annotations to the EEG data.


## Example

Parse the data.

```python
import eeg_eyetracking_parser as eet

raw, events, metadata = eet.read_subject(2)
raw.plot()
```

Plot the voltage across four occipital electrodes locked to cue onset for three seconds. This is done separately for three different conditions, defined by `cue_eccentricity`.

```python
import mne
from matplotlib import pyplot as plt

CUE_TRIGGER = 1
CHANNELS = 'O1', 'O2', 'P3', 'P4'

cue_epoch = mne.Epochs(raw, eet.epoch_trigger(events, CUE_TRIGGER), tmin=-.1,
                       tmax=3, metadata=metadata, picks=CHANNELS)
for ecc in ('near', 'medium', 'far'):
    cue_evoked = cue_epoch[f'cue_eccentricity == "{ecc}"'].average()
    plt.plot(cue_evoked.data.mean(axis=0), label=ecc)
plt.legend()
```

Plot pupil size during the same period. Because the regular `mne.Epoch()` object doesn't play nice with non-data channels, such as PupilSize, you need to use the `eet.PupilEpochs()` class instead (which is otherwise identical). You will often want to specify `reject_by_annotation=False` to avoid trials with blinks from being excluded.

```python
cue_epoch = eet.PupilEpochs(raw, eet.epoch_trigger(events, CUE_TRIGGER), tmin=0,
                       tmax=3, metadata=metadata, baseline=(0, .05), 
                       reject_by_annotation=False)
for ecc in ('near', 'medium', 'far'):
    cue_evoked = cue_epoch[f'cue_eccentricity == "{ecc}"'].average()
    plt.plot(cue_evoked.data.mean(axis=0))
plt.legend()
```

You can also convert the `PupilEpochs` object to a `SeriesColumn` and plot it that way, for example using `time_series_test.plot()`.

```python
from datamatrix import convert as cnv
import time_series_test as tst

dm = cnv.from_pandas(metadata)
dm.pupil = eet.epochs_to_series(dm, cue_epoch)
tst.plot(dm, dv='pupil', hue_factor='cue_eccentricity')
```

## Dependencies

- mne-python
- eyelinkparser


## Assumptions

### Data format

- EEG data should be in BrainVision format (`.vhdr`), recorded at 1000 Hz
- Eye-tracking data should be EyeLink format (`.edf`), recorded monocularly at 1000 Hz

### File and folder structure

Files should be organized following [BIDS](https://bids-specification.readthedocs.io/).

```
# Container folder for all data
data/
    # Subject 2
    sub-02/
        # EEG data
        eeg/
            sub-02_task-attentionalbreadth_eeg.eeg
            sub-02_task-attentionalbreadth_eeg.vhdr
            sub-02_task-attentionalbreadth_eeg.vmrk
        # Behavioral data (usually not necessary)
        beh/
            sub-02_task-attentionalbreadth_beh.csv
        # Eye-tracking data
        eye/
            sub-02_task-attentionalbreadth_physio.edf
```

### Trigger codes

The start of each trial is indicated by a counter that starts at 128 for the first trial, and wraps around after 255, such that trial 129 is indicated again by 128. This trigger does not need to be sent to the eye tracker, which uses its own `start_trial` message. A temporal offset between the `start_trial` message of the eye tracker and the start-trial trigger of the EEG is ok, and will be compensated for during parsing.

```python
EE.PulseLines(128 + trialid % 128, 10)  # EE is the EventExchange object
```

The onset of each epoch is indicated by a counter that starts at 1 for the first epoch, and then increases for subsequent epochs. In other words, if the target presentation is the second epoch of the trial, then this would correspond to trigger 2 as in the example below. This trigger needs to be sent to both the EEG and the eye tracker at the exact same moment (a temporal offset is *not* ok).

```python

target_trigger = 2
eyetracker.log(f'start_phase {target_trigger}')  # eyetracker is created by PyGaze
EE.PulseLines(target_trigger, 10)
```

Triggers should only be used for temporal information. Conditions are only logged in the eye-tracking data.


## Function reference

**<span style="color:purple">eeg&#95;eyetracking&#95;parser.epoch&#95;trigger</span>_(events, trigger)_**


Selects a single epoch trigger from a tuple with event information.
Epoch triggers have values between 1 and 127 (inclusive).


#### Parameters
* events: tuple :  Event information as returned by `read_subject()`.
* trigger: int :  A trigger code, which is a positive value.

#### Returns
<b><i>array:</i></b>  A numpy array with events as expected by mne.Epochs().



**<span style="color:purple">eeg&#95;eyetracking&#95;parser.read&#95;subject</span>_(subject_nr, folder='data/', trigger_parser=None, eeg_margin=30, min_sacc_dur=10, min_sacc_size=30, min_blink_dur=10, eye_kwargs={})_**


Reads EEG, eye-tracking, and behavioral data for a single participant.
This data should be organized according to the BIDS specification.


EEG data is assumed to be in BrainVision data format (`.vhdr`, `.vmrk`,
`.eeg`). Eye-tracking data is assumed to be in EyeLink data format (`.edf`
or `.asc`). Behavioral data is assumed to be in `.csv` format.

Metadata is taken from the behavioral `.csv` file if present, and from
the eye-tracking data if not.

#### Parameters
* subject_nr: int or sr :  The subject number to parse. If an int is passed, the subject number
	is assumed to be zero-padded to length two (e.g. '01'). If a string
	is passed, the string is used directly.
* folder: str, optional :  The folder in which the data is stored.
* trigger_parser: callable, optional :  A function that converts annotations to events. If no function is
	specified, triggers are assumed to be encoded by the OpenVibe
	acquisition software and to follow the convention for indicating
	trial numbers and event onsets as described in the readme.
* eeg_margin: int, optional :  The number of seconds after the last trigger to keep. The rest of the
	data will be cropped to save memory (in case long periods of extraneous
	data were recorded).
* min_sacc_dur: int, optional :  The minimum duration of a saccade before it is annotated as a
	BAD_SACCADE.
* min_sacc_size: int, optional :  The minimum size of a saccade (in pixels) before it is annotated as a
	BAD_SACCADE.
* min_blink_dur: int, optional :  The minimum duration of a blink before it is annotated as a
	BAD_BLINK.
* eye_kwargs: dict, optional :  Optional keyword arguments to be passed onto the EyeLink parser. If
	traceprocessor is provided, a default traceprocessor is used with
	advanced blink reconstruction enabled and 10x downsampling.

#### Returns
<b><i>tuple:</i></b>  A raw (EEG data), events (EEG triggers), metadata (a table with
	experimental variables) tuple.



**<span style="color:purple">eeg&#95;eyetracking&#95;parser.trial&#95;trigger</span>_(events)_**


Selects all trial triggers from event information. Trial triggers have
values between 128 and 255 (inclusive).


#### Parameters
* events: tuple :  Event information as returned by `read_subject()`.

#### Returns
<b><i>array:</i></b>  A numpy array with events as expected by mne.Epochs().

## License

`eeg_eyetracking_parser` is licensed under the [GNU General Public License
v3](http://www.gnu.org/licenses/gpl-3.0.en.html).
