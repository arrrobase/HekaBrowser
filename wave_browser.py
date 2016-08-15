from PyQt4 import QtGui, QtCore
from heka_reader import Bundle
import pyqtgraph as pg
import numpy as np
import spike_sort
from tqdm import tqdm
import os


class FileModel(QtGui.QFileSystemModel):
    """
    Class for file system model
    """
    def __init__(self, root_path):
        super(FileModel, self).__init__()

        # hide system files
        self.setFilter(QtCore.QDir.AllDirs |
                       QtCore.QDir.NoDotAndDotDot |
                       QtCore.QDir.AllEntries)

        # filter out non dats and disable showing
        self.setNameFilters(['*.dat'])
        self.setNameFilterDisables(False)

        # set root
        self.setRootPath(root_path)


class FileView(QtGui.QTreeView):
    """
    Class for view of file system model
    """
    def __init__(self, parent, root_path):
        super(FileView, self).__init__()
        self.frame = parent

        # set model
        self.model = FileModel(root_path)
        self.setModel(self.model)

        # set root
        self.setRootIndex(self.model.index(root_path))

        # hide unnecessary columns
        self.hideColumn(1)
        self.hideColumn(2)
        self.hideColumn(3)

        # bind event
        self.doubleClicked.connect(self.on_double_click)

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_double_click(self, index):
        """
        Event for changing .dat file
        :param index:
        :return:
        """
        model_index = self.model.index(index.row(), 0, index.parent())

        # get file from index
        file_path = os.path.abspath(self.model.filePath(model_index))

        # check extension
        _, ext = os.path.splitext(file_path)
        if ext == '.dat':
            self.frame.update_pul(file_path)


class PulView(QtGui.QTreeWidget):
    """
    Class for viewing tree of pul file
    """
    def __init__(self, parent):
        super(PulView, self).__init__()
        self.frame = parent

        self.dat_file = None
        self.bundle = None
        self.pul = None
        self.index = None

        # bind event
        self.itemSelectionChanged.connect(self.on_selection_changed)

    def update_tree(self, dat_file):
        """
        Updates tree to new pul file and sets proper dat file
        :param dat_file:
        :return:
        """
        # clear old view
        self.clear()

        # setup bundle items
        self.dat_file = dat_file
        self.bundle = Bundle(self.dat_file)
        self.pul = self.bundle.pul

        # setup view
        self.setColumnCount(2)
        self.setHeaderLabels(['Node', 'Label'])

        pulse_item = QtGui.QTreeWidgetItem(['Pulsed', ''])
        pulse_item.index = []
        self.addTopLevelItem(pulse_item)

        for i, group in enumerate(self.pul):
            group_item = QtGui.QTreeWidgetItem(pulse_item,
                                               ['Group {}'.format(i+1),
                                                group.Label])
            group_item.index = [i]
            pulse_item.addChild(group_item)

            for j, series in enumerate(group):
                series_item = QtGui.QTreeWidgetItem(group_item,
                                                    ['Series {}'.format(j+1),
                                                     series.Label])
                series_item.index = [i, j]
                group_item.addChild(series_item)

                for k, sweep in enumerate(series):
                    sweep_item = QtGui.QTreeWidgetItem(series_item,
                                                       ['Sweep {}'.format(k+1),
                                                       sweep.Label])
                    sweep_item.index = [i, j, k]
                    series_item.addChild(sweep_item)

                    for l, trace in enumerate(sweep):
                        trace_item = QtGui.QTreeWidgetItem(sweep_item,
                                                           [trace.Label])
                        trace_item.index = [i, j, k, l]
                        sweep_item.addChild(trace_item)

        # size columns
        self.expandToDepth(4)
        self.resizeColumnToContents(0)
        self.collapseAll()
        self.expandToDepth(1)

    def get_plot_params(self):
        """
        Gets information for plotting.
        :return:
        """
        if self.index is not None:
            data = self.bundle.data[self.index]

            pul = self.pul[self.index[0]][self.index[1]][self.index[2]][self.index[3]]
            y_label = pul.Label
            y_units = pul.YUnit
            x_interval = pul.XInterval

            return data, x_interval, y_label, y_units

        else:
            return None, None, None, None


    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_selection_changed(self):
        """
        Event for browsing through wave traces
        :return:
        """
        selected = self.currentItem()
        index = selected.index

        if len(index) != 4:
            return

        self.index = index

        self.frame.update_trace_plot()


class OptionsView(QtGui.QWidget):
    """
    Class for view containing plotting options.
    """
    def __init__(self, parent):
        super(OptionsView, self).__init__()
        self.frame = parent

        # init buttons
        self.extract_spikes_toggle = QtGui.QCheckBox('Extract spikes')
        self.extract_spikes_toggle.setChecked(False)

        self.group_toggle = QtGui.QCheckBox('Group spikes')
        self.group_toggle.setChecked(False)

        # layout and add
        layout = QtGui.QVBoxLayout()

        layout.addWidget(self.extract_spikes_toggle)
        layout.addWidget(self.group_toggle)
        layout.addStretch(1)

        self.setLayout(layout)

        # event binders
        self.extract_spikes_toggle.toggled.connect(self.on_extract_spikes_toggle)
        self.group_toggle.toggled.connect(self.on_group_button)

    def on_extract_spikes_toggle(self, state):
        """
        Draw extracted spikes.
        :param state:
        :return:
        """
        if state:
            if not self.frame.spike_view.isVisible():
                self.frame.spike_view.resize(self.frame.trace_view.width(),
                                             self.frame.trace_view.height()//2)
                self.frame.spike_view.show()
            self.frame.update_spike_plot()

        elif not state:
            if self.frame.spike_view.isVisible():
                self.frame.spike_view.hide()

    def on_group_button(self, state):
        """
        Toggle grouping of spikes.
        :param state:
        :return:
        """
        self.frame.toggle_grouping(state)


class PlotView(pg.PlotWidget):
    """
    Class for plot widget.
    """
    def __init__(self, parent):
        super(PlotView, self).__init__()
        self.frame = parent
        self.legend = None
        self.index = None
        self.group = False

    def plot_trace(self, data, x_interval, y_label='Record',
                   y_units='A', clear=True):
        """
        Updates plot.
        :param data:
        :param x_interval:
        :param y_label:
        :param y_units:
        :param clear:
        :return:
        """
        # make time series
        time_series, check = np.linspace(0,
                                         int(x_interval*len(data)),
                                         endpoint=False,
                                         num=len(data),
                                         retstep=True)

        # double check interval
        assert check == x_interval

        if clear:
            self.clear()

        self.plot(time_series, data)
        self.setLabels(bottom=('Time', 's'),
                       left=(y_label, y_units))

    def plot_spikes(self, data, x_interval, y_label='Record',
                    y_units='A'):
        """
        Plots spikes.
        :param data:
        :param x_interval:
        :param y_label:
        :param y_units:
        :param group:
        :return:
        """
        if self.index == self.frame.pul_view.index:
            return

        self.index = self.frame.pul_view.index
        self.clear()

        # get approximate spike times from spike_sort
        raw = {
            'data': np.array([data]),
            'FS'  : int(round(1./x_interval)),
            'n_contacts': 1
        }
        spt = spike_sort.core.extract.detect_spikes(raw, thresh='2')

        if len(spt['data']) == 0:
            if self.legend is not None:
                self.legend.scene().removeItem(self.legend)
                self.legend = None
            return 0

        if self.group:
            group = self.group_spikes(raw, spt, num_groups=2)
            assert len(group) == len(spt['data'])

        # 2.5 ms window
        ms = 2.5/1000
        half_window = int(ms / x_interval)
        spike_indices = []

        if self.group:
            running_total = np.zeros((max(group)+1, half_window*2),
                                     np.float64)
        else:
            running_total = np.zeros((half_window*2), np.float64)

        if half_window % 2 == 0:
            ms -= x_interval

        window_time_series, check = np.linspace(-ms,
                                                ms,
                                                endpoint=False,
                                                num=half_window*2,
                                                retstep=True)
        print check, x_interval, half_window
        assert check == x_interval

        # refine peak detection
        for index, time in enumerate(tqdm(spt['data'])):
            # turn time to index
            ar_index = int(time/1000*raw['FS'])
            # make window around guess
            window = data[ar_index-half_window:ar_index+half_window]
            # get index of local maximum
            peak_index = np.argmax(window) + ar_index - half_window
            # add to list
            spike_indices.append(peak_index)

            # PLOT
            # adjust window
            window = data[peak_index-half_window:peak_index+half_window]

            if self.group:
                group_ind = int(group[index])

                red = pg.mkPen(group_ind).color().red()
                blue = pg.mkPen(group_ind).color().blue()
                green = pg.mkPen(group_ind).color().green()

                self.plot(window_time_series, window,
                          pen=pg.mkPen(red, blue, green, 40))
                running_total[group_ind] += window

            else:
                self.plot(window_time_series, window, pen=pg.mkPen(200, 200, 200, 26))
                running_total += window

        if self.legend is not None:
            self.legend.scene().removeItem(self.legend)
        self.legend = self.addLegend()

        if self.group:
            for index, run in enumerate(running_total):
                run /= list(group).count(index)
                self.plot(window_time_series, run,
                          pen=pg.mkPen(255, 255, 255, 255))
        else:
            running_total /= len(spike_indices)
            self.plot(window_time_series, running_total,
                      pen=pg.mkPen(255, 255, 255, 255),
                      name='Average (n={})'.format(len(spike_indices))
                      )

        self.setLabels(bottom=('Time', 's'),
                       left=(y_label, y_units))

        return len(spike_indices)

    def group_spikes(self, raw, spt, num_groups=2):
        """
        Groups spikes.
        :param raw:
        :param spt:
        :param num_groups:
        :return:
        """
        sp_win = [-0.8, 0.8]
        spt_adj = spike_sort.core.extract.align_spikes(raw, spt, sp_win,
                                                       type='max')
        sp_waves = spike_sort.core.extract.extract_spikes(raw, spt_adj, sp_win)

        sp_feats = spike_sort.core.features.combine(
            (spike_sort.core.features.fetP2P(sp_waves),
             spike_sort.core.features.fetPCA(sp_waves))
        )

        clust_idx = spike_sort.core.cluster.cluster('gmm', sp_feats, num_groups)

        return clust_idx


class TabView(QtGui.QTabWidget):
    """
    Class for tab view.
    """
    def __init__(self, parent):
        super(TabView, self).__init__()
        self.frame = parent


class Frame(QtGui.QWidget):
    """
    Main frame.
    """
    def __init__(self):
        super(Frame, self).__init__()

        self.setWindowTitle('Wave Browser')

        root = os.path.abspath(
            'C:\\Users\\Alex\\PycharmProjects\\heka_browser\\data\\')

        # instantiate views
        self.file_view = FileView(self, root)
        self.pul_view = PulView(self)
        self.trace_view = PlotView(self)
        self.spike_view = PlotView(self)
        self.options_view = OptionsView(self)

        # make tab and add
        self.tab_view = TabView(self)
        self.tab_view.addTab(self.file_view, 'Waves')
        self.tab_view.addTab(self.options_view, 'Options')

        # layout for views
        layout = QtGui.QGridLayout()

        # splitter for pul view from tabs
        tree_splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        tree_splitter.addWidget(self.tab_view)
        tree_splitter.addWidget(self.pul_view)

        # splitter for plots
        plot_splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        plot_splitter.addWidget(self.spike_view)
        plot_splitter.addWidget(self.trace_view)
        # hide spike view until needed
        self.spike_view.hide()


        # horizontal splitter
        frame_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        frame_splitter.addWidget(tree_splitter)
        frame_splitter.addWidget(plot_splitter)

        # add to layout
        layout.addWidget(frame_splitter, 0, 0)

        # set layout and show
        self.setLayout(layout)
        self.show()

    def update_pul(self, dat_file):
        """
        Makes call to update pul view.
        :param dat_file:
        :return:
        """
        self.pul_view.update_tree(dat_file)

    def update_trace_plot(self):
        """
        Makes call to update trace plot.
        :param data:
        :param x_interval:
        :param y_label:
        :param y_units:
        :return:
        """
        data, x_interval, y_label, y_units = self.pul_view.get_plot_params()

        if data is not None:
            self.trace_view.plot_trace(data,
                                       x_interval,
                                       y_label,
                                       y_units)

            if self.spike_view.isVisible():
                self.update_spike_plot()

    def update_spike_plot(self):
        """
        Makes call to update spike plot.
        :param data:
        :param x_interval:
        :param y_label:
        :param y_units:
        :return:
        """
        data, x_interval, y_label, y_units = self.pul_view.get_plot_params()

        if data is not None:
            num_spikes = self.spike_view.plot_spikes(data,
                                                     x_interval,
                                                     y_label,
                                                     y_units)
            # print num_spikes

    def toggle_grouping(self, state):
        """
        Toggles grouping
        :param state:
        :return:
        """
        self.spike_view.group = state
        if self.spike_view.isVisible():
            self.update_spike_plot()


if __name__ == '__main__':
    app = QtGui.QApplication([])
    frame = Frame()
    # start qt event loop
    app.exec_()