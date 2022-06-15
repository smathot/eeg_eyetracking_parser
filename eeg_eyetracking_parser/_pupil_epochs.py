import mne


class PupilEpochs(mne.Epochs):
    """An Epochs class for the PupilSize channel. This allows baseline
    correction to be applied to pupil size, even though this channel is not a
    regular data channel.
    """
    
    def __init__(self, *args, **kwargs):
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = True
        super().__init__(*args, **kwargs, picks='PupilSize')
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = False
        
    def average(self, *args, **kwargs):
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = True
        evoked = super().average(*args, **kwargs)
        mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = False
        return evoked
