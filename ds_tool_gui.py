"""
GUI for direction selectivity tool.
"""

import matplotlib
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

import wx
import numpy as np

from ds_tool import get_ds


class DirPanel(wx.Panel):
    """
    Panel with options to select path to recording data and direction input.
    """
    def __init__(self, parent):
        super(DirPanel, self).__init__(parent)

        self.frame = parent.GetTopLevelParent()

        # lay out GUI elements
        path_labl = wx.StaticText(self, label='Heka data file')
        self.path_ctrl = wx.FilePickerCtrl(self,
                                           message='Path to heka data file',
                                           wildcard='*.dat',
                                           style=wx.FLP_DEFAULT_STYLE | wx.FLP_SMALL)

        init_dirs = 12
        num_dirs_labl = wx.StaticText(self, label='Number of dirs')
        num_dirs_ctrl = wx.SpinCtrl(self, min=1, initial=init_dirs, style=wx.SP_ARROW_KEYS)

        self.single_series_ctrl = wx.CheckBox(self, label='Single series')
        self.single_series_ctrl.SetValue(True)

        self.single_trace_ctrl = wx.CheckBox(self, label='Single trace')
        self.single_trace_ctrl.SetValue(True)

        self.triggered_ctrl = wx.CheckBox(self, label='Triggered')
        self.stack_ctrl = wx.CheckBox(self, label='Stack plots')

        # sizer for the above
        dirs_sizer = wx.GridBagSizer(hgap=5, vgap=5)

        dirs_sizer.Add(path_labl, pos=(0, 0), border=4, flag=wx.TOP | wx.BOTTOM)
        dirs_sizer.Add(self.path_ctrl, pos=(0, 1), border=0, flag=wx.TOP | wx.BOTTOM)

        dirs_sizer.Add(num_dirs_labl, pos=(1, 0), border=4, flag=wx.TOP | wx.BOTTOM)
        dirs_sizer.Add(num_dirs_ctrl, pos=(1, 1), border=0, flag=wx.TOP | wx.BOTTOM)

        dirs_sizer.Add(self.single_series_ctrl, pos=(2, 0), border=4, flag=wx.TOP | wx.BOTTOM)
        dirs_sizer.Add(self.single_trace_ctrl, pos=(2, 1), border=4, flag=wx.TOP | wx.BOTTOM)

        dirs_sizer.Add(self.stack_ctrl, pos=(3, 0), border=4, flag=wx.TOP | wx.BOTTOM)
        dirs_sizer.Add(self.triggered_ctrl, pos=(3, 1), border=4, flag=wx.TOP | wx.BOTTOM)

        # tracker for current "same" node location
        self.sames = [0, 0, 0]

        # dir choosers
        self.chooser_labels = []
        self.chooser_ctrls = []

        self.chooser_sizer = wx.GridBagSizer(hgap=5, vgap=5)

        # chooser title
        degrees_label = wx.StaticText(self, label='Degrees')
        degrees_series = wx.StaticText(self, label='Series')
        degrees_sweep = wx.StaticText(self, label='Sweep')
        degrees_trace = wx.StaticText(self, label='Trace')

        to_add = [degrees_series, degrees_sweep, degrees_trace]

        self.chooser_sizer.Add(degrees_label, pos=(0, 0), border=10, flag=wx.RIGHT | wx.TOP)
        for i, node in enumerate(to_add):
            self.chooser_sizer.Add(node, pos=(0, i+1), border=10, flag=wx.TOP)

        for i in range(init_dirs):
            self.add_chooser()

        # panel for whole frame
        self.panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel_sizer.Add(dirs_sizer, border=5, flag=wx.ALL)
        self.panel_sizer.Add(self.chooser_sizer, border=5, flag=wx.ALL)

        self.SetSizer(self.panel_sizer)
        self.panel_sizer.Fit(self)

        self.Bind(wx.EVT_SPINCTRL, self.on_num_dirs_ctrl, num_dirs_ctrl)

    def add_chooser(self):
        """
        Adds rows to the chooser sizer with proper control elements
        """
        num = len(self.chooser_labels)
        dirs = [0, 180, 30, 210, 60, 240, 90, 270, 120, 300, 150, 330]

        try:
            dir = dirs[num]
        except IndexError:
            dir = dirs[num % 12]

        ctrl = wx.SpinCtrl(self, min=0, max=360, initial=dir, size=(70, -1))

        self.sames[1] += 1
        nodes = []
        for i in range(3):
            node = wx.SpinCtrl(self, min=1, initial=self.sames[i], style=wx.SP_ARROW_KEYS, size=(50, -1))
            nodes.append(node)

        self.chooser_labels.append(ctrl)
        self.chooser_ctrls.append(nodes)

        self.chooser_sizer.Add(ctrl, pos=(num+1, 0), border=10, flag=wx.RIGHT)
        for i, node in enumerate(nodes):
            self.chooser_sizer.Add(node, pos=(num+1, i+1))
            self.Bind(wx.EVT_SPINCTRL, self.on_chooser_ctrl, node)
            node.tag = i

    def remove_chooser(self):
        """
        Opposite of add_chooser().
        """
        num = len(self.chooser_sizer.GetChildren())
        for i in range(1, 5):
            self.chooser_sizer.Remove(num-i)

        self.chooser_labels[-1].Destroy()
        self.chooser_labels.pop(-1)

        for node_ctrl in self.chooser_ctrls[-1]:
            node_ctrl.Destroy()
        self.chooser_ctrls.pop(-1)

        self.sames[1] = self.chooser_ctrls[-1][1].Value

    def on_num_dirs_ctrl(self, event):
        """
        Updates number of directions for analysis.
        """
        periods = event.GetInt()

        ind = len(self.chooser_labels)

        for i in range(abs(periods-ind)):

            if periods > ind:
                self.add_chooser()

            else:
                self.remove_chooser()

        # self.panel_sizer.Layout()
        self.frame.frame_sizer.Fit(self.frame)
        self.frame.Layout()

    def on_chooser_ctrl(self, event):
        """
        Keeps track of which nodes to repeat.
        """
        tag = event.GetEventObject().tag
        value = event.GetInt()

        self.sames[tag] = value

        if self.single_series_ctrl.GetValue() and tag == 0:
            for row in self.chooser_ctrls:
                row[0].SetValue(value)

        if self.single_trace_ctrl.GetValue() and tag == 2:
            for row in self.chooser_ctrls:
                row[2].SetValue(value)

    def get_nodes(self):
        """
        Makes list of nodes from controls.
        :return: list of nodes
        """
        nodes = []
        for row in self.chooser_ctrls:
            node = [0]
            for ctrl in row:
                node.append(ctrl.Value - 1)

            nodes.append(node)

        return nodes

    def get_dirs(self):
        """
        Makes list of directions.
        :return: list of directions
        """
        dirs = []
        for ctrl in self.chooser_labels:
            dirs.append(ctrl.Value)

        return dirs

    def get_triggered(self):
        """
        Returns whether or not the data has associated triggers.
        :return: bool
        """
        return self.triggered_ctrl.GetValue()

    def get_stack(self):
        """
        Returns whether or not to stack several plots.
        :return: bool
        """
        return self.stack_ctrl.GetValue()


class CalculatePanel(wx.Panel):
    """
    Panel with options to select path to recording data and direction input.
    """
    def __init__(self, parent):
        super(CalculatePanel, self).__init__(parent)

        self.frame = parent.GetTopLevelParent()

        # lay out GUI elements
        panel_sizer = wx.BoxSizer(wx.HORIZONTAL)

        calculate_button = wx.Button(self, label='Calculate')
        self.pref_dir_text = wx.StaticText(self, label='preferred direction')

        panel_sizer.Add(calculate_button, border=5, flag=wx.ALL)
        panel_sizer.Add(self.pref_dir_text, border=9, flag=wx.ALL)

        self.SetSizer(panel_sizer)
        panel_sizer.Fit(self)

        self.Bind(wx.EVT_BUTTON, self.on_calculate_button, calculate_button)

    def on_calculate_button(self, event):
        """
        Calls ds_tool with proper parameters.
        :param event:
        :return:
        """
        dat_path = self.frame.dir_panel.path_ctrl.Path
        dirs = self.frame.dir_panel.get_dirs()
        nodes = self.frame.dir_panel.get_nodes()
        triggered = self.frame.dir_panel.get_triggered()

        plt_1, plt_2, plt_3, pref_dir_degrees = get_ds(dat_path, dirs, nodes, triggered=triggered)

        pref_string = '{0:.3f}'.format(pref_dir_degrees) + u'\u00b0'
        # self.pref_dir_text.SetLabel(pref_string)

        self.frame.frame_sizer.Fit(self.frame)
        self.frame.Refresh()

        self.frame.plot_panel.draw(plt_1, plt_2, plt_3)

        # get DSI (Direction Selectivity Index)
        directions, num_spikes = plt_2
        ar_dir = np.array(directions, dtype=np.float)
        ar_resp = np.array(num_spikes, dtype=np.float)

        resp_max = np.max(ar_resp)

        resp_x = ar_resp * np.cos(ar_dir) / resp_max
        resp_y = ar_resp * np.sin(ar_dir) / resp_max

        vec_sum_x = np.sum(resp_x)
        vec_sum_y = np.sum(resp_y)

        # normalized vector sum length (i.e. DSI)
        nvsl = np.sqrt(vec_sum_x**2 + vec_sum_y**2) / np.sum(ar_resp / resp_max)
        pref_string += '   DSI: {0:.2f}'.format(nvsl)
        self.pref_dir_text.SetLabel(pref_string)


class PlotPanel(wx.Panel):
    """
    Panel with polar plot.
    """
    def __init__(self, parent):
        super(PlotPanel, self).__init__(parent)

        self.frame = parent.GetTopLevelParent()

        # lay out GUI elements
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111, projection='polar')
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def draw(self, plt_1, plt_2, plt_3):
        """
        Updates the polar plot.
        :param plt_1:
        :param plt_2:
        :param plt_3:
        """
        if not self.frame.dir_panel.get_stack():
            self.axes.clear()

        x, y_est = plt_1
        self.axes.plot(x, y_est, '-k')

        directions, num_spikes = plt_2
        self.axes.plot(directions, num_spikes, '.b')

        pref_dir, pref_fit = plt_3
        self.axes.plot(pref_dir, pref_fit, '-r')

        self.canvas.draw()


class Frame(wx.Frame):
    """
    Main frame.
    """
    def __init__(self):
        super(Frame, self).__init__(None, title='DS Tool')

        self.dir_panel = DirPanel(self)
        self.calc_panel = CalculatePanel(self)
        self.plot_panel = PlotPanel(self)

        self.dir_calc_sizer = wx.BoxSizer(wx.VERTICAL)
        self.dir_calc_sizer.Add(self.dir_panel)
        self.dir_calc_sizer.Add(self.calc_panel)

        self.frame_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.frame_sizer.Add(self.dir_calc_sizer)
        self.frame_sizer.Add(self.plot_panel)

        self.SetSizer(self.frame_sizer)
        self.frame_sizer.Fit(self)

        # change background to match panels on win32
        self.SetBackgroundColour(wx.NullColour)

        self.Show()


def main():
    """
    Main function to start GUI.
    """
    global app
    app = wx.App(False)
    frame = Frame()
    # run app
    app.MainLoop()


if __name__ == '__main__':
    main()