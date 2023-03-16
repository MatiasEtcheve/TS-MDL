from typing import Optional

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.axes import Axes
from tqdm import trange

from utils.dictionary import find_runs
from utils.eeg import VALUE_TO_CLASS

cmap = mpl.colormaps["Set2"].colors


def plot_labels(
    labels: np.ndarray,
    ax: Axes,
    times: Optional[np.ndarray] = None,
    legend: Optional[bool] = False,
) -> None:
    """Plots labels

    Args:
        labels (np.ndarray): labels to plot
        ax (Axes): Axes
        times (Optional[np.ndarray], optional): Times plotted on x axis. Defaults to None.
        legend (Optional[bool], optional): Whether to display labels. Defaults to False.
    """
    if times is None:
        times = np.arange(len(labels)) * 0.004 / 60
    run_values, run_starts, run_lengths = find_runs(labels)

    for value, start, length in zip(run_values, run_starts, run_lengths):
        if value in [1, 2, 3, 4]:
            ax.plot(
                times[int(start) : int(start + length)],
                [value] * length,
                color=cmap[value],
                label=VALUE_TO_CLASS[value]
                if (value + 6) not in labels[: start - 1]
                else None,
                linewidth=2,
            )
    ax.set_xlabel("Time (in min)")
    ax.set_yticks(
        ticks=[1, 2, 3, 4], labels=["Left hand", "Right hand", "Feet", "Tongue"]
    )
    if legend:
        ax.legend()


def save_eeg_gif(
    original: np.ndarray,
    positions: mne.Info,
    reconstruction: Optional[np.ndarray] = None,
    start: Optional[int] = 0,
    length: Optional[int] = 50,
    timeframe: Optional[int] = 1,
    fps: Optional[int] = 10,
    filename: Optional[str] = "test.gif",
    plot_errors: Optional[bool] = False,
) -> None:
    """Generates and saves a topographic video

    Args:
        original (np.ndarray): original signal of size Times x Sensors
        positions (mne.Info): positions of the EEG sensors
        reconstruction (Optional[np.ndarray]): Reconstructed signal. Useful when plotting side by side with the original signal. Defaults to None.
        start (Optional[int]): start index in the signal. Defaults to 0.
        length (Optional[int]): duration of the video (in index). Defaults to 50.
        timeframe (Optional[int]): windows of length `timeframe` is taken from the signal, and the maximum of that window is plotted. Defaults to 1.
        fps (Optional[int]): frame per second of the output. Defaults to 10.
        filename (Optional[str]): filename of the output. Defaults to "test.gif".
        plot_errors (Optional[bool]): whether to plot the error between the original and reconstructed signal. Defaults to False.
    """
    nb_plots = 1
    if reconstruction is not None:
        nb_plots += 1
        if plot_errors:
            nb_plots += 1
            errors = original - reconstruction
    fig, axs = plt.subplots(1, nb_plots, figsize=(8 * nb_plots, 8), squeeze=False)

    axs[0, 0].set_title("Original signal", fontsize=30)
    if reconstruction is not None:
        axs[0, 1].set_title("Reconstruction", fontsize=30)
        if plot_errors:
            axs[0, 2].set_title("Residual", fontsize=30)

    pbar = trange(length // timeframe, desc="Creating video from data")

    def animate(num, start):
        pbar.update(1)

        beginning = start + num * timeframe
        ending = start + (num + 1) * timeframe
        n = 25

        vlim = (
            np.min(
                original[max(beginning - n, 0) : min(start + length + n, len(original))]
            ),
            np.max(
                original[max(beginning - n, 0) : min(start + length + n, len(original))]
            ),
        )

        axs[0, 0].clear()
        mne.viz.plot_topomap(
            np.max(original[beginning:ending, :], axis=0),
            positions,
            show=False,
            axes=axs[0, 0],
            vlim=vlim,
            image_interp="linear",
        )
        axs[0, 0].set_title("Original signal", fontsize=30)
        if reconstruction is not None:
            axs[0, 1].clear()
            axs[0, 1].set_title("Reconstruction", fontsize=30)
            mne.viz.plot_topomap(
                np.max(reconstruction[beginning:ending, :], axis=0),
                positions,
                show=False,
                axes=axs[0, 1],
                vlim=vlim,
                image_interp="linear",
            )
            if plot_errors:
                axs[0, 2].clear()
                axs[0, 2].set_title("Residual", fontsize=30)
                mne.viz.plot_topomap(
                    np.max(errors[beginning:ending, :], axis=0),
                    positions,
                    show=False,
                    axes=axs[0, 2],
                    vlim=vlim,
                    image_interp="linear",
                )
        plt.tight_layout()
        return axs

    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(start,),
        interval=1 / fps * 1000,
        blit=True,
        frames=length // timeframe - 1,
    )
    ani.save(filename, dpi=50, writer=PillowWriter(fps=fps))
    pbar.close()
