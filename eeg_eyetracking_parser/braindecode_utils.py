"""A set of functions to facilitate working with braindecode"""
import random
import mne
import copy
import itertools
import logging
from collections.abc import Sequence
import collections
collections.Sequence = Sequence  # for compatibility with Skorch
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import mode
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
import torch
from datamatrix import functional as fnc, DataMatrix, operations as ops
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode.datasets import create_from_mne_epochs
from braindecode.visualization import plot_confusion_matrix
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, to_dense_prediction_model, \
    get_output_shape
from braindecode.preprocessing import Preprocessor, \
    exponential_moving_standardize
from . import read_subject, epoch_trigger

logger = logging.getLogger('eeg_eyetracking_parser')


@fnc.memoize(persistent=True)
def decode_subject(subject_nr, factors, epochs_kwargs, trigger,
                   epochs_query='practice == "no"', epochs=4, lesions=None,
                   window_size=200, window_stride=1, n_fold=4, folder='data'):
    """The main entry point for decoding a subject's data.
    
    Parameters
    ----------
    subject_nr: int
        The subject number
    factors: list of str
        A list of factors that should be decoded. Factors should be str and
        match column names in the metadata. If there is more than one factor,
        each factor should have two levels.
    epochs_kwargs: dict
        A dict with keyword arguments that are passed to mne.Epochs() to
        extract the to-be-decoded epoch.
    trigger: int
        The trigger code that defines the to-be-decoded epoch.
    epochs_query: str, optional
        A pandas-style query to select trials from the to-be-decoded epoch. The
        default assumes that there is a `practice` column from which we only
        want to have the 'no' values, i.e. that we want exclude practice
        trials.
    epochs: int, optional
        The number of training epochs, i.e. the number of times that the data
        is fed into the model
    lesions: list of tuple or str
        A list of time windows or electrode names to be set to 0 during
        testing. A separate prediction is made for each lesion. Time windows
        are (start, end) tuples in sample units. Electrode names are strings.
    window_size_samples: int
        The length of the window to sample from the Epochs object. This should
        be slightly shorter than the actual Epochs to allow for jittered
        samples to be taken from the purpose of 'cropped decoding'.
    window_stride_samples: int
        The number of samples to jitter around the window for the purpose of
        cropped decoding.
    n_fold: int
        The total number of splits (or folds)
    folder: str
        The folder in which the participant data is stored.
        
    Returns
    -------
    tuple
        A (results, metadata) tuple.
        
        
        Results is a dict where keys are labels with 'overall' corresponding to
        the overall (unlesioned) test, and the other keys corresponding to the
        lesions as provided by the lesions parameter. Values are confusion
        matrices with shape (n_conditions, n_conditions) where the first
        dimension is the real label, and the second dimension is the predicted
        label.
        
        Metadata is a DataFrame corresponding to the dataset's metadata plus
        two additional columns: braindecode_label is numeric label that encodes
        the to-be-decoded factor, and braindecode_prediction is the predicted
        label. Rows where these two columns match were correctly decoded.
    """
    dataset, labels, metadata = read_decode_dataset(
        subject_nr, factors, epochs_kwargs, trigger, epochs_query,
        window_size=window_size, window_stride=window_stride, folder=folder)
    n_conditions = len(labels)
    results = {}
    results['overall'] = np.zeros((n_conditions, n_conditions))
    predictions = DataMatrix(length=0)
    for fold in range(n_fold):
        logger.info(f'subject {subject_nr}, fold {fold}')
        train_data, test_data = split_dataset(
            dataset, fold=fold)
        clf = train(train_data, test_data, epochs=epochs)
        logger.info('Testing on complete data')
        # We want to know which trial was predicted to have which label. For
        # that reason, we create a datamatrix with true and predicted labels.
        # These are not in the original order, so we also store timestamps
        # so that later we can sort the datamatrix back into the original order
        y_pred = clf.predict(test_data)
        resized_pred = y_pred.copy()
        resized_pred.resize(
            (len(test_data.datasets), len(test_data.datasets[0])))
        fold_predictions = DataMatrix(length=len(test_data.datasets))
        fold_predictions.y_true = [d.y[0] for d in test_data.datasets]
        fold_predictions.y_pred = mode(resized_pred, axis=1)[0].flatten()
        fold_predictions.timestamp = [
            d.windows.metadata.i_start_in_trial[0]
            for d in test_data.datasets
        ]
        predictions <<= fold_predictions
        results['overall'] += summarize_accuracy(clf, test_data, factors,
                                                 labels, y_pred)
        if lesions is None:
            continue
        for lesion in lesions:
            logger.info(f'Testing on lesioned data ({lesion})')
            dataset, labels, _ = read_decode_dataset(
                subject_nr, factors, epochs_kwargs, trigger, epochs_query,
                lesion=lesion, window_size=window_size,
                window_stride=window_stride, folder=folder)
            _, lesioned_test_data = split_dataset(dataset, fold=fold)
            if lesion not in results:
                results[lesion] = np.zeros((n_conditions, n_conditions))
            results[lesion] += summarize_accuracy(
                clf, lesioned_test_data, factors, labels)
    # Add the true and predicted labels as new columns to the metadata
    predictions = ops.sort(predictions, by=predictions.timestamp)
    metadata = metadata.assign(braindecode_label=list(predictions.y_true),
                               braindecode_prediction=list(predictions.y_pred))
    return results, metadata


def read_decode_dataset(subject_nr, factors, epochs_kwargs, trigger,
                        epochs_query, lesion=None, window_size=200,
                        window_stride=1, folder='data'):
    """Reads a dataset and converts it to a format that is suitable for
    braindecode.
    """
    raw, events, metadata = _read_decode_subject(
        subject_nr, folder=folder, save_preprocessing_output=False,
        plot_preprocessing=False)
    epochs = mne.Epochs(raw, epoch_trigger(events, trigger),
                        metadata=metadata, **epochs_kwargs)
    epochs = epochs[epochs_query]
    metadata = metadata.query(epochs_query)
    if isinstance(lesion, tuple):
        epochs._data[:, :, lesion[0]:lesion[1]] = 0
    elif isinstance(lesion, str):
        epochs._data[:, epochs.ch_names.index(lesion)] = 0
    dataset, labels = build_dataset(
        epochs, metadata, factors, window_size_samples=window_size,
        window_stride_samples=window_stride)
    return dataset, labels, metadata


def _read_decode_subject(*args, **kwargs):
    """A wrapper around read_subject() that also applies
    braindecode-appropriate preprocessing.
    """
    raw, events, metadata = read_subject(*args, **kwargs)
    preprocess_raw(raw)
    return raw, events, metadata


def preprocess_raw(raw, l_freq=4, h_freq=30, factor_new=1e-3,
                   init_block_size=1000):
    """Preprocesses the raw object such that is useful for decoding. The main
    criteria seem to be that high and low frequencies are removed, and that
    the signal is normalized (0 mean and 1 std). Based on the preprocessing
    steps in the braindecode tutorials.
    """
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    Preprocessor(exponential_moving_standardize, factor_new=factor_new,
                 init_block_size=init_block_size).apply(raw)
                 

def split_epochs(epochs, metadata, factors):
    """Splits an Epochs object based on several factors, which should 
    correspond to columns in the metadata.
    
    Parameters
    ----------
    epochs: Epochs
        The Epochs object to split
    metadata: DataFrame
        The metadata that belongs to epochs
    factors: list
        A list of factors by which the data should be split.
    
    Returns
    -------
    tuple
        A tuple in which the first element is a list of Epochs objects, and the
        second element a list of str labels that define the Epochs objects.
    """
    subsets = []
    labels = []
    for code, values in enumerate(
            itertools.product(*[np.unique(metadata[f]) for f in factors])):
        subset = epochs
        label = []
        for value, factor in zip(values, factors):
            label.append(f'{factor}={value}')
            if isinstance(value, str):
                value = f"'{value}'"
            subset = subset[f"{factor} == {value}"]
        subset.events[:, 2] = code
        labels.append('\n'.join(label))
        subsets.append(subset)
    return subsets, labels


def split_dataset(dataset, n_fold=4, fold=0):
    """Split a dataset into training and testing sets using interleaved
    splitting.
    
    Parameters
    ----------
    n_fold: int
        The total number of splits (or folds)
    fold: int
        The current split number
        
    Returns
    -------
    tuple
        A (training set, test set) tuple
    """
    splitted = dataset.split({
        'train': [i for i in range(len(dataset.datasets)) if i % n_fold != fold],
        'test': [i for i in range(len(dataset.datasets)) if i % n_fold == fold]
    })
    assert len(splitted['train']) > len(splitted['test'])
    return splitted['train'], splitted['test']


def balance_dataset(dataset):
    """Makes sure that a dataset contains an equal number of observations for
    each label by randomly duplicating observations from labels that have too
    few observations. This modifies the dataset in-place.
    
    Parameters
    ----------
    dataset
    """
    n_codes = {}
    for d in dataset.datasets:
        code = d.y[0]
        if code not in n_codes:
            n_codes[code] = 0
        n_codes[code] += 1
    max_count = max(n_codes.values())
    for code, count in n_codes.items():
        n_add = max_count - count
        code_datasets = [d for d in dataset.datasets if d.y[0] == code]
        for i in range(n_add):
            dataset.datasets.append(
                copy.deepcopy(random.choice(code_datasets)))
    # We need to update these property to keep the length of the dataset
    # consistent, because they are used by the __len__() functions. Not clear
    # what the different roles of the two different properties is. See also:
    # https://github.com/pytorch/tnt/blob/master/torchnet/dataset/concatdataset.py
    dataset.cum_sizes = np.cumsum([len(x) for x in dataset.datasets])
    dataset.cumulative_sizes = dataset.cum_sizes
            

def build_dataset(epochs, metadata, factors, window_size_samples,
                  window_stride_samples):
    """Creates a dataset that is suitable for braindecode from an Epochs
    object. This indirectly implements 'cropped decoding' as explained on
    the braindecode website and accompanying paper.
    
    Parameters
    ----------
    epochs: Epochs
        The Epochs object to split
    metadata: DataFrame
        The metadata that belongs to epochs
    factors: list
        A list of factors by which the data should be split
    window_size_samples: int
        The length of the window to sample from the Epochs object. This should
        be slightly shorter than the actual Epochs to allow for jittered
        samples to be taken from the purpose of 'cropped decoding'.
    window_stride_samples: int
        The number of samples to jitter around the window for the purpose of
        cropped decoding.
    
    Returns
    -------
    tuple
        A (dataset, labels) tuple, where the labels are as returned by
        split_epochs()
    """
    epochs_list, labels = split_epochs(epochs, metadata, factors)
    dataset = create_from_mne_epochs(
        epochs_list,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False
    )
    balance_dataset(dataset)
    return dataset, labels


def train(train_set, test_set=None, epochs=4, batch_size=32, lr=0.000625,
          weight_decay=0):
    """Trains a classifier based on a training set. If a test set is provided,
    validation metrics are shown during training. However, for proper testing,
    the testing data should be predicted separately after training.

    Parameters
    ----------
    train_set: BaseDataSet
        The training data
    test_set: BaseDataSet, optional
        The testing (validation) data
    epochs: int, optional
        The number of training epochs, i.e. the number of times that the data
        is fed into the model
    batch_size: int, optional
        The number of observations that are fed into the model simultaneously
    lr: float, optional
        The learning rate. The default value is taken from the braindecode
        tutorials for the shallow network
    weight_decay: float, optional
        The weight decay. The default value is taken from the braindecode
        tutorials for the shallow network
        
    Returns
    -------
    EEGClassifier
        A trained EEG classifier
    """
    # The number of unique y (code) values
    n_classes = len(set([d.y[0] for d in train_set.datasets]))
    # Number of electrode channels is hidden deep inside the data
    n_chans = train_set.datasets[0].windows.get_data().shape[1]
    input_window_samples = train_set.datasets[0].windows.get_data().shape[2]
    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
    )
    if torch.cuda.is_available():
        model.cuda()
    to_dense_prediction_model(model)
    if test_set is None:
        train_split = None
        callbacks = None
    else:
        train_split = predefined_split(test_set)
        callbacks = [
            "accuracy", ("lr_scheduler",
                         LRScheduler('CosineAnnealingLR', T_max=epochs - 1)),
        ]
    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=train_split,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks
    )
    clf.fit(train_set, y=None, epochs=epochs)
    return clf


def build_confusion_matrix(clf, test_set, y_pred=None):
    """Builds a confusion matrix for the predicted and actual labels of a test
    set.
    
    Parameters
    ----------
    clf: EEGClassifier
        A trained EEG classifier
    test_set: BaseDataSet
        A test dataset
    y_pred: array, optional
        An array with predictions. If not provided, the predictions will be
        generated.
        
    Returns
    -------
    array
        A (n_labels, n_labels) numpy array with the true labels as the first
        axis, the predicted labels as the second axis, and the number of
        prediction in each cell as the values
    """
    y_true = []
    for d in test_set.datasets:
        y_true += d.y
    y_true = np.array(y_true)
    if y_pred is None:
        y_pred = clf.predict(test_set)
    return confusion_matrix(y_true, y_pred)


def summarize_confusion_matrix(factors, confusion_mat):
    """Determines overall classification accuracy, as well as classification
    accuracy per factor for a confusion matrix. If there is more than one
    factor, each factor is assumed to have two levels.

    Parameters
    ----------
    factors: list
        A list of factor labels
    confusion_mat: array
        A confusion matrix as returned by build_confusion_matrix()

    Returns
    -------
    list
        A list of accuracies, where the first element is the overall accuracy
        and subsequent elements correspond to the factors.
    """
    if len(factors) == 1:
        model = np.identity(confusion_mat.shape[0])
        return [100 * np.sum(model * confusion_mat) / np.sum(confusion_mat)]
    model = np.identity(2 ** len(factors))
    accuracies = []
    acc = 100 * np.sum(model * confusion_mat) / np.sum(confusion_mat)
    accuracies.append(acc)
    for i, factor in enumerate(factors):
        r1 = 2 ** (len(factors) - i - 1)
        model = np.identity(2).repeat(r1, axis=0).repeat(r1, axis=1)
        r2 = 2 ** i
        model = np.block([[model] * r2] * r2)
        acc = 100 * np.sum(model * confusion_mat) / np.sum(confusion_mat)
        accuracies.append(acc)
    return accuracies


def summarize_accuracy(clf, test_set, factors, labels, y_pred=None,
                       plot=False):
    """A convenience function built on top of build_confusion_matrix() and
    summarize_confusion_matrix() 
    
    Parameters
    ----------
    clf: EEGClassifier
        A trained EEG classifier
    test_set: BaseDataSet
        A test dataset
    factors: list
        A list of factor labels
    labels: list
        A list of labels for each combination of factors, as returned by
        split_epochs()
    y_pred: array, optional
        An array with predictions. If not provided, the predictions will be
        generated.
    plot: bool, optional
        Indicates whether a confusion-matrix plot should be created
        
    Returns
    -------
    array
        A (n_labels, n_labels) numpy array with the true labels as the first
        axis, the predicted labels as the second axis, and the number of
        prediction in each cell as the values
    """
    confusion_mat = build_confusion_matrix(clf, test_set, y_pred=y_pred)
    if plot:
        plot_confusion_matrix(confusion_mat, labels, rotate_col_labels=45,
                              rotate_row_labels=45, figsize=(12, 12))
    for factor, acc in zip(['overall'] + factors,
                           summarize_confusion_matrix(factors, confusion_mat)):
        logger.info('acc({}): {:.2f} %'.format(factor, acc))
    return confusion_mat
