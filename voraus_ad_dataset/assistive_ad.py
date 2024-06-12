"""This module contains all utility functions for the assistive-AD dataset."""

import time
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from random import sample
from typing import Dict, Generator, List, Tuple, Union

import numpy
import pandas
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def get_env_params(env_type):
    return {
        "Feeding": (25, 7),
        "Drinking": (25, 7),
        "ArmManipulation": (45, 14),
        "BedBathing": (24, 7),
        "ScratchItch": (30, 7),
    }[env_type]

def get_signals(state_size, action_size):
    class AssistiveSignals:
        """Contains the signals of the robot used in the dataset."""

        TIME = "time"
        SAMPLE = "sample"
        ANOMALY = "anomaly"
        CATEGORY = "category"
        SETTING = "setting"
        ACTION = "action"
        ACTIVE = "active"
        MDP_STATE = "mdp_state_{}"
        MDP_ACTION = "mdp_action_{}"

        MDP_STATE_SIZE = state_size
        MDP_ACTION_SIZE = action_size

        @classmethod
        def all(cls) -> tuple[str, ...]:
            return (
                (cls.TIME, cls.SAMPLE, cls.ANOMALY, cls.CATEGORY,
                 cls.SETTING, cls.ACTION, cls.ACTIVE)
                + cls.action()
                + cls.state()
            )

        @classmethod
        def state(cls) -> tuple[str, ...]:
            return tuple(cls.MDP_STATE.format(i) for i in range(cls.MDP_STATE_SIZE))

        @classmethod
        def action(cls) -> tuple[str, ...]:
            return tuple(cls.MDP_ACTION.format(i) for i in range(cls.MDP_ACTION_SIZE))

        @classmethod
        def meta(cls) -> tuple[str, ...]:
            """Returns the meta colums of the assistive-AD dataset.

            Returns:
                The meta columns of the dataset.
            """
            return (cls.TIME, cls.SAMPLE, cls.ANOMALY, cls.CATEGORY,
                    cls.SETTING, cls.ACTION, cls.ACTIVE)

        @classmethod
        def meta_constant(cls) -> tuple[str, ...]:
            """Returns time invariant meta colums of the assistive-AD dataset.

            Returns:
                The time invariant meta columns.
            """
            return (cls.SAMPLE, cls.ANOMALY, cls.CATEGORY, cls.SETTING)

        @classmethod
        def groups(cls) -> dict[str, tuple[str, ...]]:
            """Access the signal groups by name.

            Returns:
                The signal group dictionary.
            """
            class HackDict:
                def __getattribute__(self, name: str) -> time.Any:
                    return cls.state() + cls.action()
            return HackDict()
    return AssistiveSignals


class Category(IntEnum):
    """Describes the anomaly category."""
    NORMAL_OPERATION = 0
    FAILURE = 1


class Variant(IntEnum):
    """Describes the anomaly variant."""
    NO_ANOMALY = 0
    ANOMALY = 1


class Action(IntEnum):
    """Describes the actual action of the robot."""
    EXECUTE_POLICY = 0


ANOMALY_CATEGORIES = [
    Category.FAILURE,
]


@contextmanager
def measure_time(label: str) -> Generator[None, None, None]:
    """Measures the time and prints it to the console.

    Args:
        label: A label to identifiy the measured time.

    Yields:
        None.
    """
    start_time = time.time()
    yield
    print(f"{label} took {time.time()-start_time:.3f} seconds")


def extract_samples_and_labels(
    dataset: pandas.DataFrame, samples: List[int], meta_columns: List[str]
) -> Tuple[List[pandas.DataFrame], List[Dict]]:
    """Extracts one dataframe per sample from the dataset dataframe.

    Args:
        dataset: The dataset dataframe, containing all the samples.
        samples: The sample indices to extract.
        meta_columns: The meta columns to use during loading.

    Returns:
        The extracted dataframes and labels for each selected sample.
    """
    dfs = [dataset[dataset["sample"] == s].reset_index(drop=True) for s in samples]
    labels = [df.loc[0, meta_columns].to_dict() for df in dfs]
    dfs = [df.drop(columns=meta_columns) for df in dfs]
    return dfs, labels


# Disable pylint too many locals for better readability of the loading function.
def load_pandas_dataframes(  # pylint: disable=too-many-locals, too-complex
    train_path: Union[Path, str],
    test_path: Union[Path, str],
    columns: Union[List[str], Tuple],
    normalize: bool,
    frequency_divider: int,
    train_gain: float,
    pad: bool,
    env_type: str,
) -> Tuple[List[pandas.DataFrame], List[dict], List[pandas.DataFrame], List[dict]]:
    """Loads the dataset as pandas dataframes.

    Args:
        path: The path to the dataset.
        columns: The columns to load.
        normalize: Whether to normalize the data with standard scaler or not.
        frequency_divider: Scale the dataset down by dropping every nth sample.
        train_gain: The factor of train samples to use.
        pad: Whether to use zero padding or not.

    Returns:
        The dataframes and labels for each sample.
    """
    state_size, action_size = get_env_params(env_type)
    Signals = get_signals(state_size, action_size)
    if isinstance(columns, tuple):
        columns = list(columns)
    with measure_time("loading data"):
        # Add required meta columns for preprocessing
        columns_with_meta = list(columns) + list(Signals.meta_constant())
        # Read the dataset parquet file columns as pandas dataframe
        train_dataset_dataframe = pandas.read_parquet(train_path, columns=columns_with_meta)
        test_dataset_dataframe = pandas.read_parquet(test_path, columns=columns_with_meta)
        # Rename the setting column to variant
        train_dataset_dataframe.rename(columns={"setting": "variant"}, inplace=True)
        test_dataset_dataframe.rename(columns={"setting": "variant"}, inplace=True)
        # Create new meta columns list with variant inside
        meta_columns = [m for m in Signals.meta_constant() if m != "setting"] + ["variant"]

    # Check that down sampling factor is greater equal 1
    assert frequency_divider >= 1
    if frequency_divider > 1:
        with measure_time("downsampling"):
            print(f"downsample to every {frequency_divider}th frame")
            train_dataset_dataframe = train_dataset_dataframe[train_dataset_dataframe.index.values % frequency_divider == 0]
            test_dataset_dataframe = test_dataset_dataframe[test_dataset_dataframe.index.values % frequency_divider == 0]
            train_dataset_dataframe = train_dataset_dataframe.reset_index(drop=True)
            test_dataset_dataframe = test_dataset_dataframe.reset_index(drop=True)

    # select train samples
    train_idx = sorted(train_dataset_dataframe["sample"].unique())
    if train_gain < 1.0:
        with measure_time("select train samples (train gain < 1.0)"):
            train_len = len(train_idx)
            train_k = round(train_len * train_gain)
            train_idx = sample(train_idx, k=train_k)
            print(f"select {train_k} of {train_len} train samples ({train_k/train_len:.0%})")

    # load training data
    with measure_time("extract train dfs and labels"):
        dfs_train, labels_train = extract_samples_and_labels(train_dataset_dataframe, train_idx, meta_columns)

    with measure_time("extract test dfs and labels"):
        dfs_test, labels_test = extract_samples_and_labels(test_dataset_dataframe, test_dataset_dataframe["sample"].unique(), meta_columns)

    # update meta
    with measure_time("update meta data"):
        for label in labels_train + labels_test:
            label["category"] = Category(label["category"]).name
            label["variant"] = Variant(label["variant"]).name

    if normalize:
        with measure_time("normalize"):
            scale = StandardScaler()
            # using training data only
            scale.fit(pandas.concat(dfs_train))

            for df_i, dataframe in enumerate(dfs_train):
                dfs_train[df_i] = pandas.DataFrame(scale.transform(dataframe), columns=dataframe.columns)
            for df_i, dataframe in enumerate(dfs_test):
                dfs_test[df_i] = pandas.DataFrame(scale.transform(dataframe), columns=dataframe.columns)

    if pad:
        with measure_time("padding"):
            # Get maximum length from training samples
            target_length = max(len(df) for df in dfs_train)
            target_length = max(target_length, 200)
            for df_i, dataframe in enumerate(dfs_train):
                pad = pandas.DataFrame(0, index=(range(len(dataframe), target_length)), columns=dataframe.columns)
                dfs_train[df_i] = pandas.concat([dataframe, pad])
                if dfs_train[df_i].shape[1] % 2 != 0:
                    # MVT-Flow implementation does not work with odd number of signals.
                    dfs_train[df_i]['pad_dim'] = 0.0
            for df_i, dataframe in enumerate(dfs_test):
                dfs_test[df_i] = dataframe.loc[:target_length]
                pad = pandas.DataFrame(0, index=(range(len(dataframe), target_length)), columns=dataframe.columns)
                dfs_test[df_i] = pandas.concat([dataframe, pad])
                if dfs_test[df_i].shape[1] % 2 != 0:
                    # MVT-Flow implementation does not work with odd number of signals.
                    dfs_test[df_i]['pad_dim'] = 0.0

    return dfs_train, labels_train, dfs_test, labels_test


def load_numpy_arrays(
    train_path: Union[Path, str],
    test_path: Union[Path, str],
    columns: Union[List[str], Tuple],
    normalize: bool,
    frequency_divider: int,
    train_gain: float,
    pad: bool,
    env_type: str,
) -> Tuple[List[numpy.ndarray], List[dict], List[numpy.ndarray], List[dict]]:
    """Loads the dataset as numpy arrays.

    Args:
        path: The path to the dataset.
        columns: The colums to load.
        normalize: Whether to normalize the data with standard scaler or not.
        frequency_divider: Scale the dataset down by dropping every nth sample.
        train_gain: The factor of train samples to use.
        pad: Whether to use zero padding or not.

    Returns:
        The numpy arrays and labels for each sample.
    """
    x_train, y_train, x_test, y_test = load_pandas_dataframes(
        train_path=train_path,
        test_path=test_path,
        columns=columns,
        normalize=normalize,
        frequency_divider=frequency_divider,
        train_gain=train_gain,
        pad=pad,
        env_type=env_type,
    )

    x_train_arrays = [s.values for s in x_train]
    x_test_arrays = [s.values for s in x_test]

    # return shape of each array: (t, signals)
    return x_train_arrays, y_train, x_test_arrays, y_test


def load_torch_tensors(
    train_path: Union[Path, str],
    test_path: Union[Path, str],
    columns: Union[List[str], Tuple],
    normalize: bool,
    frequency_divider: int,
    train_gain: float,
    pad: bool,
    env_type: str,
) -> Tuple[List[torch.Tensor], List[dict], List[torch.Tensor], List[dict]]:
    """Loads the dataset as torch tensors.

    Args:
        path: The path to the dataset.
        columns: The colums to load.
        normalize: Whether to normalize the data with standard scaler or not.
        frequency_divider: Scale the dataset down by dropping every nth sample.
        train_gain: The factor of train samples to use.
        pad: Whether to use zero padding or not.

    Returns:
        The tensors and labels for each sample.
    """
    x_train, y_train, x_test, y_test = load_numpy_arrays(
        train_path=train_path,
        test_path=test_path,
        columns=columns,
        normalize=normalize,
        frequency_divider=frequency_divider,
        train_gain=train_gain,
        pad=pad,
        env_type=env_type
    )
    x_train_arrays = [torch.from_numpy(s).float() for s in x_train]
    x_test_arrays = [torch.from_numpy(s).float() for s in x_test]

    # return shape of each array: (t, signals)
    return x_train_arrays, y_train, x_test_arrays, y_test


class AssistiveADDataset(Dataset):
    """The assistive-AD dataset torch adapter."""

    def __init__(
        self,
        tensors: List[torch.Tensor],
        labels: List[dict],
        columns: list[str],
    ):
        """Initializes the assistive-AD dataset.

        Args:
            tensors: The tensors for each sample.
            labels: The labels for each sample.
            columns: The colums which are used.
        """
        self.tensors = tensors
        self.labels = labels
        self.columns = columns

        assert len(self.tensors) == len(self.labels), "Can not handle different label and array length."
        self.length = len(self.tensors)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return self.length

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, Dict]:
        """Access single dataset samples.

        Args:
            item: The sample index.

        Returns:
            The sample and labels.
        """
        return self.tensors[item], self.labels[item]


# Disable pylint since we need all the arguments here.
def load_torch_dataloaders(  # pylint: disable=too-many-locals
    train_dataset: Union[Path, str],
    test_dataset: Union[Path, str],
    batch_size: int,
    seed: int,
    columns: Union[List[str], Tuple],
    normalize: bool,
    frequency_divider: int,
    train_gain: float,
    pad: bool = True,
    env_type: str = "Feeding",
) -> tuple[AssistiveADDataset, AssistiveADDataset, DataLoader, DataLoader]:
    """Loads the voraus-AD dataset (train and test) as torch data loaders and datasets.

    Args:
        dataset: The path to the dataset.
        batch_size: The batch size to use.
        seed: The seed o use for the dataloader random generator.
        columns: The colums to load.
        normalize: Whether to normalize the data with standard scaler or not.
        frequency_divider: Scale the dataset down by dropping every nth sample.
        train_gain: The factor of train samples to use.
        pad: Whether to use zero padding or not.

    Returns:
        The data loaders and datasets.
    """
    x_train, y_train, x_test, y_test = load_torch_tensors(
        train_path=train_dataset,
        test_path=test_dataset,
        columns=columns,
        normalize=normalize,
        frequency_divider=frequency_divider,
        train_gain=train_gain,
        pad=pad,
        env_type=env_type,
    )

    train_dataset = AssistiveADDataset(x_train, y_train, list(columns))
    test_dataset = AssistiveADDataset(x_test, y_test, list(columns))

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_dataloader, test_dataloader
