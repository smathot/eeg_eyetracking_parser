# Python parser for combined EEG and eye-tracking data 

Copyright (2022-2024) Hermine Berberyan, Wouter Kruijne, Sebastiaan Mathôt, Ana Vilotijević


[![Publish to PyPi](https://github.com/smathot/eeg_eyetracking_parser/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/smathot/eeg_eyetracking_parser/actions/workflows/publish-package.yaml)


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
- Automated preprocessing of eye-tracking and EEG data.


## Example

Parse the data.

```python
import eeg_eyetracking_parser as eet

# eet.read_subject.clear()  # uncomment to clear the cache and reparse
raw, events, metadata = eet.read_subject(2)
raw.plot()
```

To avoid having to parse the data over and over again, `read_subject()` uses persistent [memoization](https://pydatamatrix.eu/memoization/), which is a way to store the return values of a function on disk and return them right away on subsequent calls. To clear the memoization cache, either call the `read_subject.clear()` function or remove the `.memoize` folder.

Plot the voltage across four occipital electrodes locked to cue onset for three seconds. This is done separately for three different conditions, defined by `cue_eccentricity`. The function `eet.autoreject_epochs()` behaves similarly to `mne.Epochs()`, except that autorejection is applied and that, like `read_subject()`, it uses persistent memoization.

```python
import numpy as np
import mne
from matplotlib import pyplot as plt
from datamatrix import convert as cnv

CUE_TRIGGER = 1
CHANNELS = 'O1', 'O2', 'Oz', 'P3', 'P4'

cue_epoch = eet.autoreject_epochs(raw, eet.epoch_trigger(events, CUE_TRIGGER),
                                  tmin=-.1, tmax=3, metadata=metadata,
                                  picks=CHANNELS)
```

We can convert the metadata, which is a `DataFrame`, to a `DataMatrix`, and add `cue_epoch` as a multidimensional column

```python
from datamatrix import convert as cnv
import time_series_test as tst

dm = cnv.from_pandas(metadata)
dm.erp = cnv.from_mne_epochs(cue_epoch)  # rows x channel x time
dm.mean_erp = dm.erp[:, ...]             # Average over channels: rows x time
tst.plot(dm, dv='mean_erp', hue_factor='cue_eccentricity')
```

Because the regular `mne.Epoch()` object doesn't play nice with non-data channels, such as pupil size, you need to use the `eet.PupilEpochs()` class instead. This is class otherwise identical, except that it by default removes trials where baseline pupil size is more than 2 SD from the mean baseline pupil size.

```python
pupil_cue_epoch = eet.PupilEpochs(raw, eet.epoch_trigger(events, CUE_TRIGGER),
                                  tmin=0, tmax=3, metadata=metadata,
                                  baseline=(0, .05))
dm.pupil = cnv.from_mne_epochs(pupil_cue_epoch, ch_avg=True)  # only 1 channel
tst.plot(dm, dv='pupil', hue_factor='cue_eccentricity')
```


## Installation

```
pip install eeg_eyetracking_parser
```

## Dependencies

- datamatrix >= 1.0
- eyelinkparser
- mne
- autoreject
- h5io
- braindecode
- python-picard
- json_tricks


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

You can re-organize data files into the above structure automatically with the `data2bids` command, which is part of this package. 

Assumptions:

-   all EEG files (.eeg, .vhdr, .vmrk) 
    are named in a 'Subject-00X-timestamp' format (e.g. Subject-002-[2022.06.12-14.35.46].eeg)
-   eye-tracking files (.edf)
    are named in a 'sub_X format' (e.g. sub_2.edf)
    
For example, to re-organize from participants 1, 2, 3, and 4 for a task called 'attentional-breadth', you can run the following command. This assumes that the unorganized files are in a subfolder called `data` and that the re-organized (BIDS-compatible) files are also in this subfolder, i.e. as shown above.

```
data2bids --source-path=data --target-path=data -s=1,2,3,4 -t=attentional-breadth
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
sys.path.insert(0, os.getcwd())
from npdoc_to_md import render_obj_docstring

print(render_obj_docstring(
    'eeg_eyetracking_parser.autoreject_epochs._fnc',
    'autoreject_epochs'))
print('\n\n')
print(render_obj_docstring('eeg_eyetracking_parser.epoch_trigger',
    'epoch_trigger'))
print('\n\n')
print(render_obj_docstring('eeg_eyetracking_parser.PupilEpochs',
    'PupilEpochs'))
print('\n\n')
print(render_obj_docstring('eeg_eyetracking_parser.read_subject._fnc',
    'read_subject'))
print('\n\n')
print(render_obj_docstring('eeg_eyetracking_parser.trial_trigger',
    'trial_trigger'))
print('\n\n')
print(render_obj_docstring('eeg_eyetracking_parser.braindecode_utils.decode_subject._fnc',
    'braindecode_utils.decode_subject'))
```



## License

`eeg_eyetracking_parser` is licensed under the [GNU General Public License
v3](http://www.gnu.org/licenses/gpl-3.0.en.html).
