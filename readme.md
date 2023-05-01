# Python parser for combined EEG and eye-tracking data 

Copyright (2022-2023) Hermine Berberyan, Wouter Kruijne, Sebastiaan Mathôt, Ana Vilotijević


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

## <span style="color:purple">autoreject\_epochs</span>_(\*args, ar\_kwargs=None, \*\*kwargs)_

A factory function that creates an Epochs() object, applies
autorejection, and then returns it.

__Important:__ This function uses persistent memoization, which means that
the results for a given set of arguments are stored on disk and returned
right away for subsequent calls. For more information, see
<https://pydatamatrix.eu/memoization/>

### Parameters

* **\*args: iterable**

  Arguments passed to mne.Epochs()

* **ar\_kwargs: dict or None, optional**

  Keywords to be passed to AutoReject(). If `n_interpolate` is not
  specified, a default value of [1, 4, 8, 16] is used.

* **\*\*kwargs: dict**

  Keywords passed to mne.Epochs()

### Returns

* **_Epochs:_**

  An mne.Epochs() object with autorejection applied.



## <span style="color:purple">epoch\_trigger</span>_(events, trigger)_

Selects a single epoch trigger from a tuple with event information.
Epoch triggers have values between 1 and 127 (inclusive).

### Parameters

* **events: tuple**

  Event information as returned by `read_subject()`.

* **trigger: int**

  A trigger code, which is a positive value.

### Returns

* **_array:_**

  A numpy array with events as expected by mne.Epochs().



## <span style="color:purple">PupilEpochs</span>_(\*args, baseline\_trim=(-2, 2), \*\*kwargs)_

An Epochs class for the PupilSize channel. This allows baseline
correction to be applied to pupil size, even though this channel is not a
regular data channel. In addition, this class allows pupil sizes to be
excluded based on deviant baseline values, which is recommended for
pupil analysis (but not typically done for eeg).

### Parameters

* **\*args: iterable**

  Arguments passed to mne.Epochs()

* **baseline\_trim: tuple of int, optional**

  The range of acceptable baseline values. This refers to z-scores.

* **\*\*kwargs: dict**

  Keywords passed to mne.Epochs()

### Returns

* **_Epochs:_**

  An mne.Epochs() object with autorejection applied.



## <span style="color:purple">read\_subject</span>_(subject\_nr, folder='data/', trigger\_parser=None, eeg\_margin=30, min\_sacc\_dur=10, min\_sacc\_size=100, min\_blink\_dur=10, blink\_annotation='BLINK', saccade\_annotation='SACCADE', eeg\_preprocessing=True, save\_preprocessing\_output=True, plot\_preprocessing=False, eye\_kwargs={}, downsample\_data\_kwargs={}, drop\_unused\_channels\_kwargs={}, rereference\_channels\_kwargs={}, create\_eog\_channels\_kwargs={}, set\_montage\_kwargs={}, annotate\_emg\_kwargs={}, band\_pass\_filter\_kwargs={}, autodetect\_bad\_channels\_kwargs={}, run\_ica\_kwargs={}, auto\_select\_ica\_kwargs={}, interpolate\_bads\_kwargs={})_

Reads EEG, eye-tracking, and behavioral data for a single participant.
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

### Parameters

* **subject\_nr: int or sr**

  The subject number to parse. If an int is passed, the subject number
  is assumed to be zero-padded to length two (e.g. '01'). If a string
  is passed, the string is used directly.

* **folder: str, optional**

  The folder in which the data is stored.

* **trigger\_parser: callable, optional**

  A function that converts annotations to events. If no function is
  specified, triggers are assumed to be encoded by the OpenVibe
  acquisition software and to follow the convention for indicating
  trial numbers and event onsets as described in the readme.

* **eeg\_margin: int, optional**

  The number of seconds after the last trigger to keep. The rest of the
  data will be cropped to save memory (in case long periods of extraneous
  data were recorded).

* **min\_sacc\_dur: int, optional**

  The minimum duration of a saccade before it is annotated as a
  BAD_SACCADE.

* **min\_sacc\_size: int, optional**

  The minimum size of a saccade (in pixels) before it is annotated as a
  saccade.

* **min\_blink\_dur: int, optional**

  The minimum duration of a blink before it is annotated as a blink.

* **blink\_annotation: str, optional**

  The annotation label to be used for blinks. Use a BAD_ suffix to
  use blinks a bads annotations.

* **saccade\_annotation: str, optional**

  The annotation label to be used for saccades. Use a BAD_ suffix to
  use saccades a bads annotations.

* **eeg\_preprocessing: bool or list, optional**

  Indicates whether EEG preprocessing should be performed. If `True`,
  then all preprocessing steps are performed. If a list is passed, then
  only those steps are performed for which the corresponding function
  name is in the list (e.g. `['downsample_data', 'set_montage']`)

* **save\_preprocessing\_output: bool, optional**

  Indicates whether output generated during EEG preprocessing should be
  saved.

* **plot\_preprocessing: bool, optional**

  Indicates whether plots should be shown during EEG preprocessing.

* **eye\_kwargs: dict, optional**

  Optional keyword arguments to be passed onto the EyeLink parser. If
  traceprocessor is provided, a default traceprocessor is used with
  advanced blink reconstruction enabled and 10x downsampling.

* **downsample\_data\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **drop\_unused\_channels\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **rereference\_channels\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **create\_eog\_channels\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **set\_montage\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **annotate\_emg\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **band\_pass\_filter\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **autodetect\_bad\_channels\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **run\_ica\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **auto\_select\_ica\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

* **interpolate\_bads\_kwargs: dict, optional**

  Passed as keyword arguments to corresponding preprocessing function.

### Returns

* **_tuple:_**

  A raw (EEG data), events (EEG triggers), metadata (a table with
  experimental variables) tuple.



## <span style="color:purple">trial\_trigger</span>_(events)_

Selects all trial triggers from event information. Trial triggers have
values between 128 and 255 (inclusive).

### Parameters

* **events: tuple**

  Event information as returned by `read_subject()`.

### Returns

* **_array:_**

  A numpy array with events as expected by mne.Epochs().



## <span style="color:purple">braindecode\_utils.decode\_subject</span>_(read\_subject\_kwargs, factors, epochs\_kwargs, trigger, epochs\_query='practice == "no"', epochs=4, window\_size=200, window\_stride=1, n\_fold=4, crossdecode\_read\_subject\_kwargs=None, crossdecode\_factors=None, patch\_data\_func=None, read\_subject\_func=None, cuda=True, balance=True)_

The main entry point for decoding a subject's data.

### Parameters

* **read\_subject\_kwargs: dict**

  A dict with keyword arguments that are passed to eet.read_subject() to
  load the data. Additional preprocessing as specified in
  `preprocess_raw()` is applied afterwards.

* **factors: str or list of str**

  A factor or list of factors that should be decoded. Factors should be
  str and match column names in the metadata.

* **epochs\_kwargs: dict, optional**

  A dict with keyword arguments that are passed to mne.Epochs() to
  extract the to-be-decoded epoch.

* **trigger: int**

  The trigger code that defines the to-be-decoded epoch.

* **epochs\_query: str, optional**

  A pandas-style query to select trials from the to-be-decoded epoch. The
  default assumes that there is a `practice` column from which we only
  want to have the 'no' values, i.e. that we want exclude practice
  trials.

* **epochs: int, optional**

  The number of training epochs, i.e. the number of times that the data
  is fed into the model. This should be at least 2.

* **window\_size\_samples: int, optional**

  The length of the window to sample from the Epochs object. This should
  be slightly shorter than the actual Epochs to allow for jittered
  samples to be taken from the purpose of 'cropped decoding'.

* **window\_stride\_samples: int, optional**

  The number of samples to jitter around the window for the purpose of
  cropped decoding.

* **n\_fold: int, optional**

  The total number of splits (or folds). This should be at least 2.

* **crossdecode\_read\_subject\_kwargs: dict or None, optional**

  When provided these read_subject_kwargs are passed to read_subject_func
  for reading the to-be-decoded test dataset.

* **crossdecode\_factors: str or list of str or None, optional**

  A factor or list of factors that should be decoded during tester. If
  provided, the classifier is trained using the factors specified in
  `factors` and tested using the factors specified in
  `crossdecode_factors`. In other words, specifying this keyword allow
  for crossdecoding.

* **patch\_data\_func: callable or None, optional**

  If provided, this should be a function that accepts a tuple of
  `(raw, events, metadata)` as returned by `read_subject()` and also
  returns a tuple of `(raw, events, metadata)`. This function can modify
  aspects of the data before decoding is applied.

* **read\_subject\_func: callable or None, optional**

  If provided, this should be a function that accepts keywords as
  provided through the `read_subject_kwargs` argument, and returns a
  tuple of `(raw, events, metadata)`. If not provided, the default
  `read_subject()` function is used.

* **cuda: bool, optional**

  If True, cuda will be used for GPU processing if it is available. If
  False, cuda won't be used, not even when it is available.

* **balance: bool, optional**

  Makes sure that a dataset contains an equal number of observations for
  each label by randomly duplicating observations from labels that have 
  too few observations.

### Returns

* **_DataMatrix_**

  Contains the original metadata plus four additional columns:

  - `braindecode_label` is a numeric label that corresponds to the
    to-be-decoded factor, i.e. the ground truth
  - `braindecode_prediction` is the predicted label
  - `braindecode_correct` is 1 for correct predictions and 0 otherwise
  - `braindecode_probabilities` is a SeriesColumn with the predicted
    probabilities for each label. The prediction itself corresponds to
    the index with the highest probability.

## License

`eeg_eyetracking_parser` is licensed under the [GNU General Public License
v3](http://www.gnu.org/licenses/gpl-3.0.en.html).
