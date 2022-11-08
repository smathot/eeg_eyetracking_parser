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
from datamatrix import functional as fnc, DataMatrix, operations as ops, \
    SeriesColumn, convert as cnv
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
def decode_subject(read_subject_kwargs, factors, epochs_kwargs, trigger,
                   epochs_query='practice == "no"', epochs=4, window_size=200,
                   window_stride=1, n_fold=4, crossdecode_factors=None):
    """The main entry point for decoding a subject's data.
    
    Parameters
    ----------
    read_subject_kwargs: dict
        A dict with keyword arguments that are passed to eet.read_subject() to
        load the data. Additional preprocessing as specified in
        `preprocess_raw()` is applied afterwards.
    factors: str or list of str
        A factor or list of factors that should be decoded. Factors should be
        str and match column names in the metadata.
    epochs_kwargs: dict, optional
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
        is fed into the model. This should be at least 2.
    window_size_samples: int, optional
        The length of the window to sample from the Epochs object. This should
        be slightly shorter than the actual Epochs to allow for jittered
        samples to be taken from the purpose of 'cropped decoding'.
    window_stride_samples: int, optional
        The number of samples to jitter around the window for the purpose of
        cropped decoding.
    n_fold: int, optional
        The total number of splits (or folds). This should be at least 2.
    crossdecode_factors: str or list of str, optional
        A factor or list of factors that should be decoded during tester. If
        provided, the classifier is trained using the factors specified in
        `factors` and tested using the factors specified in
        `crossdecode_factors`. In other words, specifying this keyword allow
        for crossdecoding.

    Returns
    -------
    DataMatrix
        Contains the original metadata plus four additional columns:
        
        - `braindecode_label` is a numeric label that corresponds to the
          to-be-decoded factor, i.e. the ground truth
        - `braindecode_prediction` is the predicted label
        - `braindecode_correct` is 1 for correct predictions and 0 otherwise
        - `braindecode_probabilities` is a SeriesColumn with the predicted
          probabilities for each label. The prediction itself corresponds to
          the index with the highest probability.
    """
    if not isinstance(epochs, int) or epochs < 2:
        raise ValueError('epochs should >= 2')
    if not isinstance(n_fold, int) or epochs < 2:
        raise ValueError('n_fold should >= 2')
    dataset, labels, metadata = read_decode_dataset(
        read_subject_kwargs, factors, epochs_kwargs, trigger, epochs_query,
        window_size=window_size, window_stride=window_stride)
    if crossdecode_factors is not None:
        cd_dataset, labels, metadata = read_decode_dataset(
            read_subject_kwargs, crossdecode_factors, epochs_kwargs, trigger,
            epochs_query, window_size=window_size, window_stride=window_stride)
    n_conditions = len(labels)
    predictions = DataMatrix(length=0)
    for fold in range(n_fold):
        train_data, test_data = _split_dataset(dataset, fold=fold,
                                               n_fold=n_fold)
        if crossdecode_factors is not None:
            _, test_data = _split_dataset(cd_dataset, fold=fold, n_fold=n_fold)
        clf = train(train_data, test_data, epochs=epochs)
        # We can unbalance the data after training to save time and to make the
        # cell counts match again
        _unbalance_dataset(test_data)
        # We want to know which trial was predicted to have which label. For
        # that reason, we create a datamatrix with true and predicted labels.
        # These are not in the original order, so we also store timestamps
        # so that later we can sort the datamatrix back into the original order
        y_prob = clf.predict_proba(test_data)  # probabilities for each label
        y_pred = np.argmax(y_prob, axis=1)     # the winner
        resized_pred = y_pred.copy()
        resized_pred.resize(
            (len(test_data.datasets), len(test_data.datasets[0])))
        y_prob.resize((len(test_data.datasets), len(test_data.datasets[0]),
                       n_conditions))
        fold_predictions = DataMatrix(length=len(test_data.datasets))
        fold_predictions.y_true = [d.y[0] for d in test_data.datasets]
        fold_predictions.y_pred = mode(resized_pred, axis=1)[0].flatten()
        fold_predictions.y_prob = SeriesColumn(depth=y_prob.shape[-1])
        fold_predictions.y_prob = y_prob.mean(axis=1)
        fold_predictions.timestamp = [
            d.windows.metadata.i_start_in_trial[0]
            for d in test_data.datasets
        ]
        predictions <<= fold_predictions
    # Add the true and predicted labels as new columns to the metadata
    predictions = ops.sort(predictions, by=predictions.timestamp)
    dm = cnv.from_pandas(metadata)
    dm.braindecode_label = predictions.y_true
    dm.braindecode_prediction = predictions.y_pred
    dm.braindecode_probabilities = predictions.y_prob
    dm.braindecode_correct = 0
    dm.braindecode_correct[
        dm.braindecode_label == dm.braindecode_prediction] = 1
    return dm


def read_decode_dataset(read_subject_kwargs, factors, epochs_kwargs, trigger,
                        epochs_query='practice == "no"', lesion=None,
                        window_size=200, window_stride=1):
    """Reads a dataset and converts it to a format that is suitable for
    braindecode.
    """
    raw, events, metadata = read_subject(**read_subject_kwargs)
    _preprocess_raw(raw)
    epochs = mne.Epochs(raw, epoch_trigger(events, trigger),
                        metadata=metadata, **epochs_kwargs)
    epochs = epochs[epochs_query]
    metadata = metadata.query(epochs_query)
    if isinstance(lesion, tuple):
        epochs._data[:, :, lesion[0]:lesion[1]] = 0
    elif isinstance(lesion, str):
        epochs._data[:, epochs.ch_names.index(lesion)] = 0
    dataset, labels = _build_dataset(
        epochs, metadata, factors, window_size_samples=window_size,
        window_stride_samples=window_stride)
    return dataset, labels, metadata


def train(train_set, test_set=None, epochs=4, batch_size=32, lr=0.000625,
          weight_decay=0, predict_nonlinearity='auto'):
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
    predict_nonlinearity: str, None, or callable
        See EEGClassifier docs
        
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
        logger.info('enabling cuda for gpu acceleration')
        model.cuda()
    else:
        logger.info('cuda is not available, not enabling gpu acceleration')
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
        callbacks=callbacks,
        predict_nonlinearity=predict_nonlinearity
    )
    clf.fit(train_set, y=None, epochs=epochs)
    return clf


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


def build_confusion_matrix(labels, predictions):
    """Creates a confusion matrix based on the results of `decode_subject()`.
    
    Parameters
    ----------
    labels: DataMatrix column
        A column with true labels (`dm.braindecode_label`)
    predictions: DataMatrix column
        A column with predicted labels or probabilities for all labels
        (`dm.braindecode_prediction` or `dm.braindecode_probabilities`). 
    
    Returns
    -------
    array:
        A confusion matrix with the true labels as the first axis and the
        predicted labels (or summed probabilities of the predicted labels) as
        the second axis.
    """
    if hasattr(predictions, 'depth'):
        cm = np.zeros((labels.count, labels.count), dtype=float)
        for label, prediction in zip(labels, predictions):
            cm[label] += prediction
    else:
        cm = np.zeros((labels.count, labels.count), dtype=int)
        for label, prediction, ldm in ops.split(labels, predictions):
            cm[label, prediction] = len(ldm)
    return cm


def _preprocess_raw(raw, l_freq=4, h_freq=30, factor_new=1e-3,
                   init_block_size=1000):
    """Preprocesses the raw object such that is useful for decoding. The main
    criteria seem to be that high and low frequencies are removed, and that
    the signal is normalized (0 mean and 1 std). Based on the preprocessing
    steps in the braindecode tutorials.
    """
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    Preprocessor(exponential_moving_standardize, factor_new=factor_new,
                 init_block_size=init_block_size).apply(raw)
                 

def _split_epochs(epochs, metadata, factors):
    """Splits an Epochs object based on several factors, which should
    correspond to columns in the metadata.

    Parameters
    ----------
    epochs: Epochs
        The Epochs object to split
    metadata: DataFrame
        The metadata that belongs to epochs
    factors: str or list
        A factor or list of factors by which the data should be split.

    Returns
    -------
    tuple
        A tuple in which the first element is a list of Epochs objects, and the
        second element a list of str labels that define the Epochs objects.
    """
    if isinstance(factors, str):
        factors = [factors]
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


def _split_dataset(dataset, n_fold=4, fold=0):
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
    return splitted['train'], splitted['test']


def _balance_dataset(dataset):
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
            copied_dataset = copy.deepcopy(random.choice(code_datasets))
            copied_dataset._is_balance_copy = True
            dataset.datasets.append(copied_dataset)
        if n_add:
            logger.info(
                f'adding {n_add} observations to code {code} to balance data')
    _update_dataset_size(dataset)


def _update_dataset_size(dataset):
    """This updates properties of the dataset to keep the length consistent,
    because they are used by the __len__() functions. Not clear what the
    different roles of the two different properties is. See also:
    
    - https://github.com/pytorch/tnt/blob/master/torchnet/dataset/concatdataset.py
    
    This modifies the dataset in-place.

    Parameters
    ----------
    dataset
    """
    dataset.cum_sizes = np.cumsum([len(x) for x in dataset.datasets])
    dataset.cumulative_sizes = dataset.cum_sizes


def _unbalance_dataset(dataset):
    """Removes copied datasets that were add during balancing of the data. This
    modifies the dataset in-place.

    Parameters
    ----------
    dataset
    """
    to_remove = []
    for d in dataset.datasets:
        if hasattr(d, '_is_balance_copy') and d._is_balance_copy:
            to_remove.append(d)
    if not to_remove:
        return
    logger.info(f'removing {len(to_remove)} observations to unbalance data')
    for d in to_remove:
        dataset.datasets.remove(d)
    _update_dataset_size(dataset)


def _build_dataset(epochs, metadata, factors, window_size_samples,
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
    factors: str or list
        A factor or list of factors by which the data should be split.
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
    epochs_list, labels = _split_epochs(epochs, metadata, factors)
    dataset = create_from_mne_epochs(
        epochs_list,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False
    )
    _balance_dataset(dataset)
    return dataset, labels
