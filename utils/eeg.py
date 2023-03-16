from os import listdir
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

from utils.dictionary import numpy_fill

SENSOR_TO_EEG_NAMES = {
    "Fz": "EEG-Fz",
    "FC3": "EEG-0",
    "FC1": "EEG-1",
    "FCz": "EEG-2",
    "FC2": "EEG-3",
    "FC4": "EEG-4",
    "C5": "EEG-5",
    "C3": "EEG-C3",
    "C1": "EEG-6",
    "Cz": "EEG-Cz",
    "C2": "EEG-7",
    "C4": "EEG-C4",
    "C6": "EEG-8",
    "CP3": "EEG-9",
    "CP1": "EEG-10",
    "CPz": "EEG-11",
    "CP2": "EEG-12",
    "CP4": "EEG-13",
    "P1": "EEG-14",
    "Pz": "EEG-Pz",
    "P2": "EEG-15",
    "POz": "EEG-16",
}

VALUE_TO_CLASS = {1: "Left hand", 2: "Right hand", 3: "Feet", 4: "Tongue"}


def get_custom_montage(
    sensor_to_eeg_names: Optional[Dict] = SENSOR_TO_EEG_NAMES,
) -> mne.channels.DigMontage:
    """Generate the montage of the experiment

    Args:
        sensor_to_eeg_names (Optional[Dict]): Mapping the dataset sensors to usual names of the sensors. Defaults to SENSOR_TO_EEG_NAMES.

    Returns:
        mne.channels.DigMontage: montage of the dataset
    """
    montage = "standard_1020"
    dig_montage = mne.channels.make_standard_montage(montage)
    standard_positions = dig_montage.get_positions()

    filtered_positions = {
        "ch_pos": {
            sensor_to_eeg_names[sensor_name]: position
            for sensor_name, position in standard_positions["ch_pos"].items()
            if sensor_name in sensor_to_eeg_names.keys()
        },
        "coord_frame": standard_positions["coord_frame"],
        "nasion": standard_positions["nasion"],
        "lpa": standard_positions["lpa"],
        "rpa": standard_positions["rpa"],
        "hsp": standard_positions["hsp"],
        "hpi": standard_positions["hpi"],
    }

    custom_montage = mne.channels.make_dig_montage(**filtered_positions)
    return custom_montage


def clean_raw(raw: mne.io.Raw, custom_montage: mne.channels.DigMontage) -> mne.io.Raw:
    """Cleans a raw EEG. Especially, it splits EEG and EOG channels and set the montage of the EEG.

    Args:
        raw (mne.io.Raw): data to clean
        custom_montage (mne.channels.DigMontage): montage to set

    Returns:
        mne.io.Raw: cleaned EEG
    """
    all_sensor_names = [channel["ch_name"] for channel in raw.info["chs"]]
    raw.set_channel_types(
        {
            sensor_name: "eeg" if "EEG" in sensor_name else "eog"
            for sensor_name in all_sensor_names
        }
    )
    raw.set_montage(custom_montage)


def get_annotations(raw: mne.io.Raw) -> np.ndarray:
    """Retrieves all the annotations from an EEG. We only keep the cues.

    Args:
        raw (mne.io.Raw): EEG

    Returns:
        np.ndarray: annotations. vector of length Times
    """
    events, _ = mne.events_from_annotations(raw, verbose=False)
    all_labels = np.empty(len(raw.get_data()[0]))
    all_labels[:] = np.nan
    all_labels[events[:, 0]] = events[:, 2]
    all_labels = numpy_fill(all_labels)

    idx = np.isin(all_labels - 6, list(VALUE_TO_CLASS.keys()))
    labels = np.zeros(len(all_labels), dtype=int)
    labels[idx] = (all_labels[idx] - 6).astype(int)
    return labels


def load_dataset(
    path_to_data: Optional[str] = "data/BCICIV_2a_gdf/",
    custom_montage: Optional[mne.channels.DigMontage] = None,
    preprocessing: Optional[bool] = True,
) -> Tuple[List[mne.io.Raw], List[np.ndarray], List[mne.io.Raw], float]:
    """Load a whole dataset

    Args:
        path_to_data (Optional[str]): path to the dataset. Must be a folder containing the .gdf files. Defaults to "data/BCICIV_2a_gdf/".
        custom_montage (Optional[mne.channels.DigMontage]): Montage of the dataset. Defaults to None (ie takes a standard 10-20).
        preprocessing (Optional[bool]): whether to filter the data. Defaults to True.

    Returns:
        Tuple[List[mne.io.Raw], List[np.ndarray], List[mne.io.Raw], float]: Tuple containing:
            * a list of training raw EEG
            * a list of labels observed during the training
            * a list of testing raw EEG
            * a sampling period (should be 0.004s)
    """
    if custom_montage is None:
        custom_montage = get_custom_montage()

    train_data = []
    train_labels = []
    test_data = []

    Ts = None

    items = listdir(path_to_data)
    for item in items:
        # Load session
        raw = mne.io.read_raw_gdf(path_to_data + item, verbose=False, preload=True)

        # Apply bandwith filter
        if preprocessing:
            raw = raw.filter(l_freq=8, h_freq=30, picks="all", verbose=False)

        # Convert to ndarray and discard EOG channels
        clean_raw(raw, custom_montage)
        labels = get_annotations(raw)

        train_flag = item.split(".")[0][-1] == "T"
        if train_flag:
            train_data.append(raw)
            train_labels.append(labels)
        else:
            test_data.append(raw)

        if Ts is None:
            Ts = raw["EEG-Fz"][1][1]
    return train_data, train_labels, test_data, Ts
