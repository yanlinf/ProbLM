"""
Plot memory usage of different models.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

LABELS = ['CountMinSketch', 'Naive', 'CountSketch']
COLORS = ['r--', 'g', 'b--']


def plot_memory_usage(fig, ax, filename):
    """
    Plot the memory usage of a model on the given axis.

    fig: plt.Figure object
    ax: plt.matplotlib.axes.Axes object
    filename: str

    Returns: None
    """
    with open(filename, 'r', encoding='utf-8') as fin:
        fin.readline()  # skip the first line
        table = [line.split()[1:3] for line in fin]

    table = np.array(table, dtype=np.float64)

    y, x = table[:, 0], table[:, 1]
    x = x - x.min()
    x = x / x.max() * 2000

    ax.plot(x, y, COLORS.pop(), label=LABELS.pop())


def main(args):
    fig, ax = plt.subplots()

    for filename in args.infiles:
        plot_memory_usage(fig, ax, filename)

    ax.set_xlabel('number of processed lines')
    ax.set_ylabel('memory used (in MiB)')
    ax.grid()
    ax.legend()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles',
                        type=str,
                        nargs='+',
                        help='.dat files that contains the memory usage data')
    args = parser.parse_args()
    main(args)
