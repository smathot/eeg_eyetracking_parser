# Python parser for combined EEG and eye-tracking data 

Copyright (2022) Hermine Berberyan, Wouter Kruijne, Sebastiaan Mathôt, Ana Vilotijević

## Table of contents

- [About](#about)
- [Example](#example)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Assumptions](#assumptions)
- [Function reference](#function-reference)
- [License](#license)

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

To avoid having to parse the data over and over again, you can use [memoization](https://pydatamatrix.eu/memoization/), which is a way to store the return values of a function:

```python
from datamatrix import functional as fnc


@fnc.memoize(persistent=True)
def read_subject(subject_nr):
    return eet.read_subject(subject_nr)


# read_subject.clear()  # uncomment to clear the cache and reparse
raw, events, metadata = read_subject(2)
```


Plot the voltage across four occipital electrodes locked to cue onset for three seconds. This is done separately for three different conditions, defined by `cue_eccentricity`.

```python
import mne
from matplotlib import pyplot as plt

CUE_TRIGGER = 1
CHANNELS = 'O1', 'O2', 'P3', 'P4'

cue_epoch = mne.Epochs(raw, eet.epoch_trigger(events, CUE_TRIGGER), tmin=-.1,
                       tmax=3, metadata=metadata, picks=CHANNELS,
                       reject_by_annotation=False)
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

## Installation

```
pip install eeg_eyetracking_parser
```

## Dependencies

- mne-python
- eyelinkparser
- autoreject


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
        eyetracking/
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

``` { .python silent }
import sys, os
sys.path.append(os.getcwd())
import eeg_eyetracking_parser as eet
from npdoc_to_md import render_md_from_obj_docstring

print(render_md_from_obj_docstring(eet.epochs_to_series, 'eeg_eyetracking_parser.epochs_to_series'))
print('\n\n')
print(render_md_from_obj_docstring(eet.epoch_trigger, 'eeg_eyetracking_parser.epoch_trigger'))
print('\n\n')
print(render_md_from_obj_docstring(eet.PupilEpochs, 'eeg_eyetracking_parser.PupilEpochs'))
print('\n\n')
print(render_md_from_obj_docstring(eet.read_subject, 'eeg_eyetracking_parser.read_subject'))
print('\n\n')
print(render_md_from_obj_docstring(eet.trial_trigger, 'eeg_eyetracking_parser.trial_trigger'))
```



## License

`eeg_eyetracking_parser` is licensed under the [GNU General Public License
v3](http://www.gnu.org/licenses/gpl-3.0.en.html).
