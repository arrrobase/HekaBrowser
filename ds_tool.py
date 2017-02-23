"""
Tool to find the preferred direction of a DS cell from Heka data
"""
# from matplotlib import pyplot as plt
from heka_reader import Bundle

import numpy as np
import operator
import math


def get_spike_indices(trace, edge, num_stds):
    """
    Gets indices of spikes in extracellular recordings.
    Adapted from Bartosz Telenczuk: https://github.com/btel/SpikeSort
    """
    thresh = np.std(trace) * num_stds

    op1, op2 = operator.lt, operator.gt

    edges = ['rising', 'falling']

    assert edge in edges
    if edge == 'falling':
        op1, op2, = op2, op1
        thresh = -thresh

    i, = np.where(op1(trace[:-1], thresh) & op2(trace[1:], thresh))

    return i


def fit_sine_polar(x, y):
    """
    Fits a sine wave with a frequency of 1 to data from a polar plot.
    Adapted from Ed Tate: http://exnumerus.blogspot.com/2010/04/how-to-fit-sine-wave-example-in-python.html
    """
    rows = [[np.sin(t), np.cos(t), 1] for t in x]

    a = np.matrix(rows)
    b = np.matrix(y).T

    w = np.linalg.lstsq(a, b)[0]

    phase = math.atan2(w[1, 0], w[0, 0])
    amplitude = np.linalg.norm([w[0, 0], w[1, 0]], 2)
    bias = w[2, 0]

    return phase, amplitude, bias


def get_ds(dat_path, directions, nodes, direction='falling', triggered=False):
    """
    Gets the preferred direction of the cell.

    :param dat_path: Heka .dat file
    :param directions: list of directions, in degrees
    :param nodes: list of nodes
    :param direction: whether wave  index is on rising or falling edge
    :param triggered: whether or not there is associated trigger data

    :return: preferred direction, in degrees
    """
    assert len(directions) == len(nodes)
    assert direction in ['falling', 'rising']

    bundle = Bundle(dat_path)
    data = bundle.data

    num_series = len(bundle.pul[0][nodes[0][1]])
    nodes = nodes[:num_series]
    directions = directions[:num_series]

    onsets = None
    if triggered:
        # get index of trigger spikes
        traces = [data[[a, b, c, 2]] for a, b, c, d in nodes]
        onsets = [get_spike_indices(trace, 'rising', 15)[-2] for trace in traces]

    # get spike indices, starting at trigger onset
    traces = [data[node] for node in nodes]
    if onsets is not None:
        traces = [trace[onset:] for trace, onset in zip(traces, onsets)]

    spike_indices = [get_spike_indices(trace, direction, 4) for trace in traces]
    num_spikes = [len(spike_index) for spike_index in spike_indices]

    directions = np.array(directions) / 180. * math.pi

    # plt.polar(directions, num_spikes, '.r')

    phase, amplitude, bias = fit_sine_polar(directions, num_spikes)

    x = np.arange(0, 2 * math.pi + 0.05, 0.05)
    y_est = amplitude * np.sin(x + phase) + bias

    plt_1 = (x, y_est)
    plt_2 = (directions, num_spikes)

    # plt.polar(x, y_est, '-k')
    # plt.polar(directions, num_spikes, '.b')

    pref_dir = math.pi / 2 - phase
    if pref_dir < 0:
        pref_dir += 2 * math.pi
    pref_fit = amplitude * math.sin(pref_dir + phase) + bias

    plt_3 = ([pref_dir, 0], [pref_fit * 1.2, 0])
    # plt.polar([pref_dir, 0], [pref_fit * 1.2, 0], '-r')
    # plt.show()

    pref_dir_degrees = pref_dir / math.pi * 180
    return plt_1, plt_2, plt_3, pref_dir_degrees


if __name__ == '__main__':
    pass