import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import colorConverter
from colorsys import rgb_to_hls, hls_to_rgb

from collections import OrderedDict, Iterable

from . import utils
log = utils.get_plain_logger(__name__)


class PlotManager:
    def __init__(self, legend_loc='upper right'):
        self.view = None    # current view
        self.views = {}     # task: view
        self.configs = {}   # task: [(cls, args), ...]
        self.labels = {}    # view, row, col, 'x/y': 'xylabel', kwargs
        self.active = {}    # view: [loc, ...]
        self.windows = {}   # window: None or figure

        # used for caching
        self.lms = {}       # loc: task: config-idx: lm
        self.axes = {}      # loc: (row, col): ax
        self.lines = {}     # line: lm
        self.tasks = {}     # view: [task, ...]

        # ToDo: change axes for current view
        self.enable(None)

        self.legend_loc = legend_loc
        self.selection = dict()

    def _reset(self):
        self.lms = {}       # loc: task: [lm, ...]
        # ToDo:             # loc: task: config-idx: lm
        self.axes = {}      # loc: (row, col): ax
        self.lines = {}     # line: lm
        self.tasks = {}     # view: [task, ...]
        self.selection = {}
        for num, fig in self.windows.items():
            if self.is_window_open(num):
                fig.clf()
                fig.canvas.draw()

    def new_view(self):
        return max(self.views.values(), default=-1) + 1

    def new_window(self):
        num = max(self.windows.keys(), default=-1) + 1
        self.windows[num] = None
        return num

    def enable(self, view_or_task=None, window=None):
        if view_or_task in self.views:
            view = self.views[view_or_task]
        else:
            view = view_or_task
        if window is None or window not in self.windows:
            window = self.new_window()
        loc = window, 0, 0
        self.active.setdefault(view, []).append(loc)
        # logging
        win_label = self.get_window_label(window)
        if view is None:
            view_label = 'current view'
        else:
            view_label = 'view {}'.format(view)
        log.info('{} enabled on {}'.format(view_label, win_label))

        if view is None:
            view = self.view
        tasks = self.tasks.get(view, set())
        for task in tasks:
            log.info('update view-loc {} with task: {}'.format(loc, task))
            self._update_loc(loc, task)
            # activate all axes from current view
            try:
                if self.axes[loc]:
                    for ax in self.axes[loc].values():
                        ax.relim()
                        ax.autoscale_view()
                        ax.set_visible(1)
                    # Set View Title
                    fig = ax.get_figure()
                    #~ fig = self.open_window(loc[0])
                    _tasks = []
                    for _task in self.lms[loc]:
                        if self.views[_task] == view:
                            _tasks.append(repr(_task))
                    s = 'View {}: {}'.format(view, ', '.join(_tasks))
                    fig.suptitle(s)
            except KeyError:
                pass
        if tasks:
            self.draw()


    def disable(self, view_or_task=None, window=-1):
        if view_or_task in self.views:
            view = self.views[view_or_task]
        else:
            view = view_or_task
        if view in self.active:
            self.tasks.pop(view, None)
            # logging
            if view is None:
                view_label = 'current view'
            else:
                view_label = 'view {}'.format(view)
            # remove view locations
            locs = []
            if window == -1:
                    locs.append(self.active[view][-1])
            elif window is None:
                locs += self.active[view]
            else:
                for loc in self.active[view]:
                    if loc[0] == window:
                        locs.append(loc)
            for loc in locs:
                self.active[view].remove(loc)
                win_label = self.get_window_label(loc[0])
                log.info('{} removed from {}'.format(view_label, win_label))
                self.lms.pop(loc, None)
            if not self.active[view]:
                self.active.pop(view)
                log.info('{} is disabled'.format(view_label))

            used_locs = set(e for es in self.active.values() for e in es)
            for loc in locs:
                window = loc[0]
                if loc not in used_locs:
                    if self.is_window_open(window):
                        label = self.get_window_label(window)
                        log.info('close {}'.format(label))
                        plt.close(label)
                        self.windows.pop(window)

    def join_views(self, *tasks):
        if len(tasks) > 1:
            newview = self.views[tasks[0]]
            for task in tasks[1:]:
                oldview = self.views[task]
                if oldview == newview:
                    continue
                self.views[task] = newview
                # check xylabels
                keys = [key for key in self.labels if key[0] == oldview]
                for oldkey in keys:
                    _oldview, row, col, xy = oldkey
                    newkey = newview, row, col, xy
                    label, unit, kwargs = self.labels.pop(oldkey)
                    msg = 'move {}label={!r} of row={}, col={} on ' \
                          'view {} into new view {}'
                    msg = msg.format(xy, label, row, col, _oldview,
                                     newview)
                    log.debug(msg)
                    if newkey not in self.labels and (label or unit):
                        self.labels[newkey] = label, unit, kwargs
                    elif xy == 'y' and unit == self.labels[newkey][1] and \
                         (label or unit):
                        self.labels[newkey] = '', unit, kwargs
                    elif unit != self.labels[newkey][1]:
                        msg = 'view={}, row={}, col={}: try to set ' \
                              '{}label={!r} / {!r} but {!r} / {!r} ' \
                              'is used.'
                        msg = msg.format(
                            row, col, newview,
                            xy, label, unit,
                            self.labels[newkey][0],
                            self.labels[newkey][1],
                        )
                        log.error(msg)
                        #~ raise ValueError(msg)

    def plot(self, task, x='', y='', squeeze='', accumulate=None,
             row=0, col=0, use_cursor=True, **kwargs):
        if task not in self.views:
            self.views[task] = self.new_view()
        view = self.views[task]

        #~ if view not in self.active:
            #~ self.enable(view)

        xpath = task._get_argpath(x)
        if xpath:
            xlabel = self._path_to_label(xpath, task)
            xunit = xpath[-1]._unit
            self.xlabel(task, xlabel, xunit, row, col)
        else:
            self.xlabel(task, 'index', '', row, col)
        ypath = task._get_argpath(y)
        yunit = ypath[-1]._unit
        if len(ypath) == 1:
            ylabel = self._path_to_label(ypath, task).replace('$', '')
        else:
            ylabel = ''
        self.ylabel(task, ylabel, yunit, row, col)

        args = dict(
            xpath=xpath,
            ypath=ypath,
            squeeze=squeeze,
            accumulate=accumulate,
            use_cursor=use_cursor,
            kwargs=kwargs,
        )
        config = row, col, LineManager, args
        self.configs.setdefault(task, []).append(config)
        #~ self.update(task)

    def xlabel(self, task, label, unit=None, row=0, col=0, **kwargs):
        if task not in self.views:
            self.views[task] = self.new_view()
        view = self.views[task]
        key = view, row, col, 'x'
        if key not in self.labels and label:
            self.labels[key] = label, unit, kwargs
            msg = '{!r}: set xlabel={!r} on axes row={}, col={} of view={}'
            msg = msg.format(task, label, row, col, view)
            log.debug(msg)
        elif label and label != self.labels[key][0]:
            msg = 'try to set xlabel={!r} but axes on row={} and col={} ' \
                  'of view {} already has a xlabel={!r}'
            msg = msg.format(label, row, col, view, self.labels[key][0])
            log.error(msg)
            #~ raise ValueError(msg)

    def ylabel(self, task, label, unit=None, row=0, col=0, **kwargs):
        if task not in self.views:
            self.views[task] = self.new_view()
        view = self.views[task]
        if 'rotation' not in kwargs:
            kwargs['rotation'] = 'horizontal'
        if 'horizontalalignment' not in kwargs or 'ha' not in kwargs:
            kwargs['ha'] = 'right'
        if 'labelpad' not in kwargs:
            kwargs['labelpad'] = 0
        key = view, row, col, 'y'
        if (label or unit) and key not in self.labels:
            self.labels[key] = label, unit, kwargs
        elif (label or unit) and unit == self.labels[key][1]:
            self.labels[key] = '', unit, kwargs
        elif (label or unit) and unit != self.labels[key][1]:
            msg = 'view={}, row={}, col={}: try to set ' \
                  'ylabel={!r} / {!r} but {!r} / {!r} ' \
                  'is used.'
            msg = msg.format(
                view, row, col,
                label, unit,
                self.labels[key][0],
                self.labels[key][1],
            )
            #~ msg = 'try to set ylabel={!r} / {!r} but axes on ' \
                  #~ 'row={} and col={} of view {} already has a ' \
                  #~ 'ylabel={!r} / {!r}'
            #~ msg = msg.format(label, unit, row, col, view,
                #~ self.labels[key][0], self.labels[key][1])
            log.error(msg)
            #~ raise ValueError(msg)

    def _update_loc(self, loc, task):
        """update view location loc with data from task"""
        newlines = {}
        for idx, (row, col, cls, args) in enumerate(self.configs[task]):
            lm = self.lms.get(loc, {}).get(task, {}).get(idx, None)
            if lm:
                lines = lm.update()
                newlines.setdefault(lm, []).extend(lines)
            elif task.clen:
                view = self.views[task]
                ax = self.get_axis(view, loc, row, col)
                lm = cls(loc, ax, task, self, **args)
                confs = self.lms.setdefault(loc, OrderedDict())
                confs.setdefault(task, {})[idx] = lm
                lines = lm.update()
                newlines.setdefault(lm, []).extend(lines)
                try:
                    ylabel, yunit, args = self.labels[view, row, col, 'y']
                    if yunit:
                        if ylabel:
                            fmt = r'$\dfrac{{{}}}{{\mathsf{{{}}}}}$'
                            ylabel = fmt.format(ylabel, yunit)
                        else:
                            ylabel = '[{}]'.format(yunit)
                    ax.set_ylabel(ylabel, **args)
                except KeyError:
                    pass
        #~ for lm, lines in newlines.items():
            #~ for line in lines:
                #~ self.lines[line] = lm

    def update(self, task, enable_cursor=True):
        """update data from task on all active view locations"""
        if task.clen < 1:
            return
        locs = []
        if task in self.views:
            view = self.views[task]
            locs += self.active.get(view, [])
            locs += self.active.get(None, [])

        view_tasks = {task}
        if view != self.view and None in self.active:
            if view_tasks is not None:
                view_tasks.update(self.tasks.get(view, []))
            msg = 'change current view from {} to {}'
            msg = msg.format(self.view, view)
            log.info(msg)
            for loc in self.active[None]:
                self.axes.pop(loc, None)
                confs_by_task = self.lms.get(loc, {})
                for conf in confs_by_task.values():
                    for lm in conf.values():
                        for line in lm.lines.values():
                            self.lines.pop(line)
                    conf.clear()
                fig = self.open_window(loc[0])
                fig.clf()
                fig.canvas.draw()
                msg = '    cleared figure {} from view loc {}'
                msg = msg.format(fig.number, loc)
                log.info(msg)
        self.view = view

        for loc in locs:
            tasks = set(view_tasks)
            if self.is_window_closed(loc[0]):
                log.info('window {} was closed'.format(loc[0]))
                self.axes.pop(loc, None)
                confs_by_task = self.lms.get(loc, {})
                for conf in confs_by_task.values():
                    for lm in conf.values():
                        for line in lm.lines.values():
                            self.lines.pop(line)
                    conf.clear()
                tasks.update(self.tasks.get(view, []))
            for t in tasks:
                log.info('update view-loc {} with task: {}'.format(loc, t))
                self._update_loc(loc, t)
                self.tasks.setdefault(view, set()).add(t)

            # activate all axes from current view
            try:
                if self.axes[loc]:
                    for ax in self.axes[loc].values():
                        ax.relim()
                        ax.autoscale_view()
                        ax.set_visible(1)
                    # Set View Title
                    fig = ax.get_figure()
                    #~ fig = self.open_window(loc[0])
                    _tasks = []
                    for _task in self.lms[loc]:
                        if self.views[_task] == view:
                            _tasks.append(repr(_task))
                    s = 'View {}: {}'.format(view, ', '.join(_tasks))
                    fig.suptitle(s)
            except KeyError:
                pass
        if not enable_cursor:
            for lm in self.lms[loc][task].values():
                if lm.cursor is not None:
                    lm.cursor.set_visible(False)
                leg = lm.ax.get_legend()
                if leg is not None:
                    leg.remove()
        self.draw()

    def draw(self):
        for window, fig in self.windows.items():
            if self.is_window_open(window):
                fig = self.open_window(window)
                self.tight_layout(fig)
                fig.canvas.draw()
                fig.show()
                #~ fig.canvas.flush_events()

    def get_axis(self, view, loc, row, col):
        try:
            return self.axes[loc][row, col]
        except KeyError:
            ax = self.create_axis(view, loc, row, col)
            loc_axes = self.axes.setdefault(loc, {})
            loc_axes[row, col] = ax
            return ax

    def create_axis(self, view, loc, row, col):
        try:
            axes = list(self.axes[loc].values())
            ax = axes[0]
            rows, cols = ax.get_subplotspec().get_gridspec().get_geometry()
        except KeyError:
            axes = []
            rows, cols = 1, 1
        ridx = self._loc2idx(row)
        nrows = max(rows, 1 if ridx[1] is None else ridx[1])
        cidx = self._loc2idx(col)
        ncols = max(cols, 1 if cidx[1] is None else cidx[1])
        gs = gridspec.GridSpec(nrows, ncols)
        if nrows > rows or ncols > cols:
            for ax in axes:
                x, y = self._xypos(rows, cols, ax.get_subplotspec().num1)
                num1 = self._posxy(x, y, nrows, ncols)
                num2 = ax.get_subplotspec().num2
                if num2 is not None:
                    x, y = self._xypos(rows, cols, num2)
                    num2 = self._posxy(x, y, nrows, ncols)
                # from ax.change_geometry()
                ax._subplotspec = gridspec.SubplotSpec(gs, num1, num2)
                ax.update_params()
                ax.set_position(ax.figbox)
        subspec = gs[slice(*ridx), slice(*cidx)]
        fig = self.open_window(loc[0])
        fig.canvas.set_window_title(self.get_window_title(loc[0]))
        ax = fig.add_subplot(subspec)
        try:
            xlabel, xunit, kwargs = self.labels[view, row, col, 'x']
            if xunit:
                xlabel += ' / {}'.format(xunit)
            ax.set_xlabel(xlabel, **kwargs)
        except KeyError:
            pass
        # ylabel in _update()
        self.tight_layout(fig)
        return ax

    def tight_layout(self, fig):
        # decrease rect area due to fig.suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    def get_window_label(self, window):
        return 'PM-Window {}'.format(window)

    def get_window_title(self, window):
        views = []
        for view, locs in self.active.items():
            for loc in locs:
                if loc[0] == window:
                    if view is None:
                        views.append('Current View')
                    else:
                        views.append('View {}'.format(view))
        views = ' and '.join(sorted(views))
        if views:
            return '{} on PM-Window {}'.format(views, window)
        else:
            return 'PM-Window {}'.format(window)

    @staticmethod
    def _loc2idx(loc):
        if isinstance(loc, (tuple, list)):
            if loc[1] is not None:
                return (loc[0], loc[1]+1)
            else:
                return (loc[0], loc[1])
        elif loc is None:
            return (None, None)
        else:
            return (loc, loc+1)

    @staticmethod
    def _xypos(rows, cols, pos):
        row = pos // cols
        col = pos % cols
        return row, col

    @staticmethod
    def _posxy(row, col, nrows, ncols):
        return row*ncols + col

    def open_window(self, window):
        label = self.get_window_label(window)
        fig = plt.figure(label)
        if fig is not self.windows[window]:
            fig.clf()
            ids = fig.canvas.callbacks.callbacks.get('pick_event', [])
            for id in tuple(ids):
                fig.canvas.mpl_disconnect(id)
            fig.canvas.mpl_connect('pick_event', self.on_pick)
            self.windows[window] = fig
        return fig

    def is_window_open(self, window):
        if window not in self.windows:
            return False
        elif self.windows[window] is None:
            return False
        elif plt.fignum_exists(self.windows[window].number):
            return True
        else:
            return False

    def is_window_closed(self, window):
        return not self.is_window_open(window)

    @staticmethod
    def _path_to_label(path, task):
        if path and len(task.depend_tasks.get(path[-1]._task, [])) < 2:
            _path = path[-1:]
        else:
            _path = path
        label = []
        for param in _path:
            name = param.name.replace('_', '\_')
            if param._task is task:
                s = r'${}$'.format(name)
            else:
                s = r'${}_\mathsf{{{}}}$'
                s = s.format(name, param._task.name)
            label.append(s)
        return ' | '.join(label)


    def on_pick(self, event):
        idx = event.ind[0]
        line = event.artist
        #~ line.figure.canvas.flush_events()
        lmngr = self.lines[line]
        line.figure.canvas.flush_events()

        #~ xdata = line.get_xdata()
        #~ ydata = line.get_ydata()
        #~ lmngr.set_cursor(xdata, ydata, line.get_color(), idx)

        cidxs = lmngr.cidxs[line]
        try:
            cidx = cidxs[idx]
        except:
            cidx = cidxs[0]

        legs = {}
        for task, confs in self.lms[lmngr.loc].items():
            if task is lmngr.task:
                for lm in confs.values():
                    ln, yval = lm.update_cursor(cidx)
                    legs.setdefault(lm.ax, []).append([ln, lm, yval])
            else:
                for lm in confs.values():
                    if lm.cursor is not None:
                        lm.cursor.set_visible(False)

        for ax in self.axes[lmngr.loc].values():
            leg_lines = []
            leg_labels = []
            for ln, lm, yval in legs.get(ax, []):
                ylabel = self._path_to_label(lm.ypath, lm.task)
                if yval is not None:
                    ylabel += ' = {}'.format(yval)
                leg_lines.append(ln)
                leg_labels.append(ylabel)
            if ax is line.axes:
                arg_lines = []
                for name, arg in lmngr.task.args:
                    if len(arg._states) > 1:
                        _name = name.replace('_', '\_')
                        val = arg.get_cache(cidx)
                        if isinstance(val, Iterable) and not \
                           isinstance(val, str):
                            label = r'${} = \ldots$'.format(_name)
                        else:
                            label = r'${} = {:.7g}$'.format(_name, val)
                        arg_lines.append(label)
                leg_lines.append(line)
                leg_labels.append('\n'.join(arg_lines))
                leg = ax.legend(leg_lines, leg_labels,
                                loc=self.legend_loc)

                arg_line = leg.legendHandles[-1]
                arg_line.set_visible(False)
                arg_line._legmarker.set_visible(False)
                view = self.views[lmngr.task]
                if len(self.tasks[view]) > 1:
                    leg.set_title(lmngr.task.name)
            elif leg_lines:
                leg = ax.legend(leg_lines, leg_labels,
                                loc=self.legend_loc)
                view = self.views[lmngr.task]
                if len(self.tasks[view]) > 1:
                    leg.set_title(lmngr.task.name)
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
        fig = line.get_figure()
        fig.canvas.draw()
        #~ fig.canvas.flush_events()

        self.selection = dict(task=lmngr.task,
                              cidx=cidx,
                              _lm=lmngr,
                              _line=line,
                              _idx=idx)


class LineManager:
    def __init__(self, loc, ax, task, pm, xpath, ypath,
                 squeeze, accumulate, use_cursor,
                 kwargs):
        self.loc = loc
        self.ax = ax
        self.task = task
        self.use_cursor = use_cursor
        self.kwargs = kwargs
        self.pm = pm
        self.xpath = xpath
        self.ypath = ypath

        # inverse of self.task.nested_args
        levels = {}
        for arg, level in self.task.args._nested_args.items():
            levels.setdefault(level, []).append(arg)
        # squeeze
        all_args = set()
        for arg in self.task.args._get(squeeze):
            if arg in self.task.args._nested_args:
                level = self.task.args._nested_args[arg]
                all_args.update(levels[level])
        self.squeeze = all_args
        # accumulate
        if accumulate is None and self.squeeze:
            accumulate = '*'
        if accumulate == '*':
            args = [arg for n, arg in self.task.args]
        else:
            args = self.task.args._get(accumulate)
        all_args = set(args)
        for arg in args:
            if arg in self.task.args._nested_args:
                level = self.task.args._nested_args[arg]
                all_args.update(levels[level])
        self.mask = [arg in all_args for n, arg in self.task.args]
        if not self.squeeze and not any(self.mask):
            self.use_cursor = False

        self.lines = {}     # key: line
        self.cidxs = {}     # line: [cidx, ...]
        self.clen = 0

        self.cursor = None

    def get_key(self, cidx):
        # ToDo: results could be cached, maybe in self._keys (cidx: key)
        key = []
        for name, arg in self.task.args:
            if arg in self.squeeze:
                key.append(None)
            else:
                key.append(arg._cache[cidx])
        return tuple(key)

    def update(self):
        datas = OrderedDict()   # key: cidxs, xvals, yvals
        for cidx in range(self.clen, self.task.clen):
            key = self.get_key(cidx)
            cidxs, xvals, yvals = datas.setdefault(key, ([], [], []))
            cidxs.append(cidx)
            if self.xpath:
                values = self.task.get_value_from_path(self.xpath, cidx)
                xvals.append(values)
            values = self.task.get_value_from_path(self.ypath, cidx)
            yvals.append(values)

        ydata = []
        newlines = []
        for key, (cidxs, xvals, yvals) in datas.items():
            if key not in self.lines:
                line = self.newline()
                self.lines[key] = line
                self.cidxs[line] = cidxs
                newlines.append(line)
                self.pm.lines[line] = self
            else:
                line = self.lines[key]
                self.cidxs[line].extend(cidxs)

            ydata = line.get_ydata()
            ydata = np.append(ydata, yvals)
            line.set_ydata(ydata)

            if xvals:
                xdata = line.get_xdata()
                xdata = np.append(xdata, xvals)
            else:
                xdata = np.arange(len(ydata))
            line.set_xdata(xdata)

        if len(ydata):
            #~ self.set_cursor(xdata, ydata, line.get_color())
            self.update_cursor(cidx)


        self.clen = self.task.clen

        # set/reset visibility of lines
        if self.lines:
            cidx = self.clen - 1
            self.show_accumulate_lines(cidx)

        return newlines

    def newline(self):
        kwargs = dict(self.kwargs)
        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'
        #~ line, = self.ax.plot([], [], picker=10, **kwargs)
        line, = self.ax.plot([], [], **kwargs)
        return line

    def show_accumulate_lines(self, cidx):
        state = [arg._cache[cidx] for n, arg in self.task.args]
        #~ state = self.get_key(cidx)
        for key, line in self.lines.items():
            value = all(self.compare(key, state))
            line.set_visible(value)
            line.set_picker(10 if value else None)

    def compare(self, states, other):
        for s1, s2, m in zip(states, other, self.mask):
            if None in (s1, s2):
                yield True
            else:
                yield m or s1 == s2

    def update_cursor(self, cidx):
        key = self.get_key(cidx)
        line = self.lines[key]
        cidxs = self.cidxs[line]

        idx = cidxs.index(cidx)
        xdata, ydata = line.get_data()
        yval = self.set_cursor(xdata, ydata, line.get_color(), idx)
        self.show_accumulate_lines(cidx)
        return line, yval

    def set_cursor(self, xdata, ydata, color, idx=-1, cidx=None):
        if not self.use_cursor:
            return
        if self.squeeze or len(ydata) == 1:
            xval = xdata[idx]
            yval = ydata[idx]
            if self.cursor is None:
                self.cursor, = self.ax.plot([], [],
                    marker='o',
                    markersize=14,
                    alpha=0.5,
                    zorder=1,
                )
        else:
            marker = None
            xval = xdata
            yval = ydata
            if self.cursor is None:
                self.cursor, = self.ax.plot([], [],
                    linewidth=7,
                    marker=None,
                    markersize=14,
                    alpha=0.5,
                    zorder=1,
                )
        #~ if clen is None:
            #~ cidx = self.clen - 1
        #~ yval = self.task.get_value_from_path(self.ypath, cidx)

        self.cursor.set_data([xval, yval])
        #~ self.cursor.set_color(shade_color(line.get_color(), 50))
        self.cursor.set_color(color)
        self.cursor.set_visible(True)
        return yval if self.squeeze or len(ydata) == 1 else None


    @staticmethod
    def _shade_color(color, percent):
        """ A color helper utility to either darken or lighten given color

        from https://github.com/matplotlib/matplotlib/pull/2745
        """
        rgb = colorConverter.to_rgb(color)
        h, l, s = rgb_to_hls(*rgb)
        l *= 1 + float(percent)/100
        l = np.clip(l, 0, 1)
        r, g, b = hls_to_rgb(h, l, s)
        return r, g, b


if __name__ == '__main__':
    import doctest
    doctest.testmod(
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    utils.enable_logger(log, format='short')

    from looping import Task

    @Task
    def sigma():
        value = 0.1
        return value
    sigma.run()

    @Task
    def offs(x, sigma=0.1):
        y = x * np.random.normal(1, sigma)
        return y

    offs.args.x.sweep(0, 5)
    offs.args.sigma.depends_on(sigma.returns.value)
    offs.run(0)
    offs.run(0)
    offs.run(0)

    @Task
    def t1(x: 'cm'):
        y: 'kg' = (x**3 - 0.5*x) * 10
        return y
    t1.args.x.sweep(-1, 1, num=9)
    t1.run()

    @Task
    def t2(x: 'cm', m=1, offs=0):
        y: 'kg' = m*x + offs
        return y
    t2.args.m.value = 10
    t2.args.x.sweep(-1, 1, num=5)
    t2.run()

    @Task
    def t3(x: 'cm', m=0.1, offs: 'V' = 0, values=[]):
        y: 'V' = m * x**2 + offs
        _x = 0.8*np.array([0, 1, 1, 0, 0]) + x
        _y: 'V' = 4*np.array([0, 0, 1, 1, 0]) + offs
        return y, _x, _y
    t3.args.x.sweep(0, 5)
    #~ t3.args.offs.iterate(0, 10, 20)
    t3.args.offs.depends_on(t2.returns.y, sweeps='x')
    t3.args.values.depends_on(t2.returns.y, squeeze='x', sweeps='x')
    t3.args.zip('offs', 'values')
    t3.run()

    #~ t3.run(0)
    #~ t3.run()

    pm = PlotManager()

    pm.plot(t1, x='x', y=t1.returns.y, squeeze=t1.args.x)
    pm.plot(t2, x='x', y=t2.returns.y, squeeze='x')
    pm.plot(t2, x='x', y=t2.returns.y, squeeze='x', row=1)
    pm.join_views(t1, t2)
    #~ pm.xlabel('x', t1)
    #~ pm.xlabel('x', t1)
    #~ pm.ylabel('y', t1)
    #~ pm.xlabel('x', t2, row=1)

    pm.plot(t3, y=t3.returns.y, squeeze='x')
    #~ pm.plot(t3, y=t3.args.offs, squeeze='x')
    #~ pm.xlabel('index', t3)

    #~ pm.plot(t3, x='x', y=t3.args.offs, squeeze='x', col=1)
    #~ pm.plot(t3, x='x', y=t2.returns.y, squeeze='x', col=1)
    pm.plot(t3, x='x', y=[t2.args.x, 'offs'], squeeze='x', col=1)

    pm.plot(t3, x='x', y=t3.returns.y, row=1, col=0, color='r')
    pm.plot(t3, x='_x', y='_y', row=1, col=1, accumulate='offs')

    #~ pm.disable()
    #~ pm.enable(0)
    #~ pm.enable(1)

"""
    pm.plot(task1, 'x', 'y', squeeze='x', col=1)
    pm.plot(task1, 'x', 'z', squeeze='x', row=2)

    pm.plot(task2, 'x', 'out', squeeze='x', row=None)
    pm.join_views(task1, task2)
"""
