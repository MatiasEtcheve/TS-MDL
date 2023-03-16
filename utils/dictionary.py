import pickle
from typing import Tuple

import numpy as np

from mdla.mdla import MultivariateDictLearning


def save_dictionary(dico: MultivariateDictLearning, filename: str) -> None:
    """
    Saves a dictionary into a pkl file
    """
    filehandler = open(filename, "wb")
    pickle.dump(dico, filehandler)
    filehandler.close()


def load_dictionary(filename: str) -> MultivariateDictLearning:
    """
    Loads a dictionary from a pkl file
    """
    file = open(filename, "rb")
    object_file = pickle.load(file)
    file.close()
    return object_file


def find_runs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find runs of consecutive items in an array

    Raises:
        ValueError: if the array is more than 1D

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing
            * the values of each run
            * the start index of each run
            * the length of each run
    """
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def numpy_fill(x: np.ndarray) -> np.ndarray:
    """
    Imitates the method pandas.ffill
    """
    mask = np.isnan(x)
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = x[idx]
    return out


def reconstruct_signal_from_code(
    code: np.ndarray, dico: MultivariateDictLearning, sample_signal: np.ndarray
) -> np.ndarray:
    """Reconstruct signal from code

    Args:
        code (np.ndarray): list of arrays (n_nonzero_coefs, 3)
            The sparse code decomposition: (amplitude, offset, kernel)
            for all n_nonzero_coefs. The list concatenates all the
            decomposition of the n_samples of input X
        dico (MultivariateDictLearning): dictionary
        sample_signal (np.ndarray): sample signal to infer the number of channels and the length

    Returns:
        np.ndarray: reconstructed signal
    """
    reconstructions = []
    n = len(sample_signal)
    for code in code:
        signal = np.zeros((n, sample_signal.shape[1]))
        for i, c in enumerate(code):
            atom = np.zeros((n, sample_signal.shape[1]))
            scale, shift, kernel = c[0], int(c[1]), int(c[2])
            l, v = dico.kernels_[int(kernel)].shape
            atom[shift : shift + l, :] = dico.kernels_[kernel]
            signal += scale * atom
        reconstructions.append(signal)
    reconstructions = np.array(reconstructions)
    return reconstructions


def compute_reconstruction_rate(
    original: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    """Computes the reconstruction rate, on axis = 0

    Args:
        original (np.ndarray): original signal, of size Records x Time x Channel
        reconstruction (np.ndarray): reconstruction, of size Records x Time x Channel

    Returns:
        np.ndarray: vector of length Records
    """
    return 1 - np.mean(
        np.linalg.norm(reconstruction - original, axis=1)
        / np.linalg.norm(original, axis=1),
        axis=-1,
    )


def compute_mse(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """Computes the mean squared error, on axis = 0

    Args:
        original (np.ndarray): original signal, of size Records x Time x Channel
        reconstruction (np.ndarray): reconstruction, of size Records x Time x Channel

    Returns:
        np.ndarray: vector of length Records
    """
    return np.mean((original - reconstruction) ** 2, axis=(1, 2))


def gabor_dict_1d(
    n: int, kmax: float, fmin: float, fmax: float, alpha: float, beta: float
) -> np.ndarray:
    """Generates a 1D Gabor dictionary of size n

    Args:
        n (int): the size of the dictionary
        kmax (float): the maximum frequency of the Gabor function
        fmin (float): the minimum frequency of the Gabor function
        fmax (float): the maximum frequency of the Gabor function
        alpha (float): the scaling factor of the Gaussian envelope
        beta (float): the orientation of the Gabor function

    Returns:
        np.ndarray: a dictionary of size n containing Gabor functions
    """

    D = np.zeros(
        (
            n,
            len(beta)
            * len(list(np.arange(fmin, fmax, (fmax - fmin) / n)))
            * len(alpha),
        )
    )

    ind = 0
    for b in beta:
        for f in np.arange(fmin, fmax, (fmax - fmin) / n):
            for a in alpha:
                k = kmax * np.cos(b)
                sigma = kmax / (2 * np.pi * f)
                x = np.arange(-n // 2, n // 2)
                G = np.exp(-np.pi * (a**2) * (x**2) / (sigma**2))
                G = G * np.exp(1j * 2 * np.pi * k * x)
                D[:, ind] = np.real(G)
                ind += 1

    return D
