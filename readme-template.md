# Python parser for combined EEG and eye-tracking data 

Copyright (2022) Hermine Berberyan, Wouter Kruijne, Sebastiaan Mathôt, Ana Vilotijević

## About

A Python module for reading concurrently recorded EEG and eye-tracking data, and parsing this data into convenient objects for further analysis. For this to work, several assumptions need to be met, as described under [Assumptions](#assumptions). At present, this module is largely for internal use, and focused on our own recording environment.

Key features:

- Experimental variables (such as conditions) from the eye-tracking data are used as metadata for the EEG analysis.
- Gaze and pupil data is added as channels to the EEG data. (This data is also included in the `DataMatrix` object, where it is split into phases/ epochs.)
- Blinks and saccades from the eye-tracking data are added as BAD annotations to the EEG data.


## Example

Parse the data.

```python
import eeg_eyetracking_parser as eet

raw, events, metadata, dm = eet.read_subject(2)
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

Plot pupil size during the same period. We can do this in two ways. First, we can use the 'PupilSize' channel and plot it similar to how we plot the EEG data:

```python
cue_epoch = mne.Epochs(raw, eet.epoch_trigger(events, CUE_TRIGGER), tmin=0,
                       tmax=1, metadata=metadata, baseline=None, preload=True)
for ecc in ('near', 'medium', 'far'):
    cue_evoked = cue_epoch[f'cue_eccentricity == "{ecc}"'].average(picks='PupilSize')
    plt.plot(cue_evoked.data.mean(axis=0))
plt.legend()
```

We can also use the `dm` object. In this case we need to concatenate phases 1 and 2, because phase 1 is by itself only 1000 ms (i.e. trigger 2 was sent after 1000 ms).

```python
from datamatrix import series as srs
import time_series_test as tst

dm.pupil = srs.concatenate(srs.endlock(dm.ptrace_1), dm.ptrace_2)
dm.pupil.depth = 300
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

``` { .python silent }
import sys
sys.path.append('/home/sebastiaan/git/eeg_eyetracking_parser')
import eeg_eyetracking_parser as eet
from npdoc_to_md import render_md_from_obj_docstring

print(render_md_from_obj_docstring(eet.epoch_trigger, 'eeg_eyetracking_parser.epoch_trigger'))
print('\n\n')
print(render_md_from_obj_docstring(eet.read_subject, 'eeg_eyetracking_parser.read_subject'))
print('\n\n')
print(render_md_from_obj_docstring(eet.trial_trigger, 'eeg_eyetracking_parser.trial_trigger'))
```



## License

`eeg_eyetracking_parser` is licensed under the [GNU General Public License
v3](http://www.gnu.org/licenses/gpl-3.0.en.html).
