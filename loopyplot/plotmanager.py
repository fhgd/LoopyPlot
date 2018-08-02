import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import colorConverter
from colorsys import rgb_to_hls, hls_to_rgb

from collections import OrderedDict, Iterable
from itertools import product

from . import utils
from . import taskmanager
log = utils.get_plain_logger(__name__)


class PlotManager:
    def __init__(self, legend_loc='upper right'):
        self.view = None    # current view
        self.views = {}     # task: view
        self.configs = {}   # task: [(cls, args), ...]
        self.labels = {}    # view, row, col, 'x/y': 'xylabel', kwargs
        self.xlim = {}      # view, row, col: (xmin, xmax)
        self.active = {}    # view: [loc, ...]
        self.windows = {}   # window: None or figure
        self.xyparams = {}   # view, row, col: [params]

        # used for caching
        self.lms = {}       # loc: task: config-idx: lm
        self.axes = {}      # loc: (row, col): ax
        self.lines = {}     # line: lm
        self.tasks = {}     # view: [task, ...]
        self.ax_params = {} # param: ax
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
            for task in tasks:
                if task not in self.views:
                    self.views[task] = self.new_view()
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

                # join xparams
                keys = [key for key in self.xyparams if key[0] == oldview]
                for oldkey in keys:
                    _oldview, row, col, xy = oldkey
                    newkey = newview, row, col, xy
                    params = self.xyparams.pop(oldkey)
                    self.xyparams.setdefault(newkey, []).extend(params)

                # join xlims
                keys = [key for key in self.xlim if key[0] == oldview]
                for oldkey in keys:
                    _oldview, row, col = oldkey
                    if _oldview != oldview:
                        continue
                    xmin, xmax = self.xlim.pop(oldkey)
                    newkey = newview, row, col
                    if newkey not in self.xlim:
                        self.xlim[newkey] = xmin, xmax
                    else:
                        _xmin, _xmax = self.xlim[newkey]
                        _xlim = min(xmin, _xmin), max(xmax, _xmax)
                        self.xlim[newkey] = _xlim

    def plot(self, task, x=[], y=[], squeeze=[], accumulate='*',
             row=0, col=0, use_cursor=True, xsort=None, **kwargs):
        if task not in self.views:
            self.views[task] = self.new_view()
        view = self.views[task]

        #~ if view not in self.active:
            #~ self.enable(view)

        if isinstance(row, list):
            row = tuple(row)
        if isinstance(col, list):
            col = tuple(col)

        xpath = task.complete_path(x)
        if xpath != x:
            msg = ['xpath is completed: {}',
                   '              from: {!r}']
            msg = '\n'.join(msg).format(xpath, x)
            log.debug(msg)
        if xpath:
            xlabel = self._path_to_short_label(xpath, task)
            xunit = xpath[-1]._unit
            self.xlabel(task, xlabel, xunit, row, col)
        else:
            self.xlabel(task, 'index', '', row, col)
        ypath = task.complete_path(y)
        if ypath != y:
            msg = ['ypath is completed: {}',
                   '              from: {!r}']
            msg = '\n'.join(msg).format(ypath, y)
            log.debug(msg)
        yunit = ypath[-1]._unit
        if len(ypath) == 1:
            ylabel = self._path_to_short_label(ypath, task)
            self.ylabel(task, ylabel, yunit, row, col)
        else:
            ylabel = self._path_to_short_label(ypath, task)
            self.ylabel(task, ylabel, yunit, row, col,
                        rotation='vertical',
                        horizontalalignment='center')

        _squeeze = squeeze
        if squeeze is None and (xpath in task.args._get_arg_paths() or
                                xpath and xpath[-1] in task.args
        ):
            squeeze = [xpath]
        elif squeeze != '*':
            if not isinstance(squeeze, (list, tuple)):
                squeeze = [squeeze]
            squeeze = [task.complete_path(path) for path in squeeze]
        if squeeze != _squeeze:
            msg = ['squeeze is completed: {}',
                   '                from: {!r}']
            msg = '\n'.join(msg).format(squeeze, _squeeze)
            log.debug(msg)

        if squeeze and xpath and hasattr(xpath[-1], '_ptrs'):
                xmin = xpath[-1]._ptrs.min
                xmax = xpath[-1]._ptrs.max
                key = view, row, col
                if key not in self.xlim:
                    self.xlim[key] = xmin, xmax
                else:
                    _xmin, _xmax = self.xlim[key]
                    self.xlim[key] = min(xmin, _xmin), max(xmax, _xmax)

        if accumulate != '*':
            if not isinstance(accumulate, (list, tuple)):
                accumulate = [accumulate]
            accumulate = [task.complete_path(path) for path in accumulate]

        if xsort is None:
            xsort = False
            if xpath and hasattr(xpath[-1], '_ptrs'):
                pointers = xpath[-1]._ptrs._pointers
                Concat = taskmanager.Concat
                if any(isinstance(p.sweep, Concat) for p in pointers):
                    xsort = True

        args = dict(
            xpath=xpath,
            ypath=ypath,
            squeeze=[] if squeeze is None else squeeze,
            accumulate=accumulate,
            use_cursor=use_cursor,
            xsort=xsort,
            kwargs=kwargs,
        )
        config = row, col, LineManager, args
        self.configs.setdefault(task, []).append(config)

        # set xyparams
        if xpath:
            key = view, row, col, 'x'
            self.xyparams.setdefault(key, []).append(xpath[-1])
        key = view, row, col, 'y'
        self.xyparams.setdefault(key, []).append(ypath[-1])

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
            kwargs['horizontalalignment'] = 'right'
        if 'labelpad' not in kwargs:
            kwargs['labelpad'] = 0
        key = view, row, col, 'y'
        if (label or unit) and key not in self.labels:
            self.labels[key] = label, unit, kwargs
            log.debug('key is new: {}'.format(key))
        elif (label or unit) and unit == self.labels[key][1]:
            self.labels[key] = '', unit, kwargs
            log.debug('key has same units but different labels')
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
        if task not in self.configs:
            return
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
                            if args['rotation'] == 'horizontal':
                                ylabel = ylabel.replace('$', '')
                                fmt = r'$\dfrac{{{}}}{{\mathsf{{{}}}}}$'
                            else:
                                fmt = r'{} / {}'
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
        else:
            return

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
                self.windows[loc[0]] = None
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
            xparams = self.xyparams.get((view, row, col, 'x'), [])
            xparam = xparams[0] if len(xparams) == 1 else None
            xax = self.ax_params.get((xparam, 'x'), None)

            yparams = self.xyparams.get((view, row, col, 'y'), [])
            yparam = yparams[0] if len(yparams) == 1 else None
            yax = self.ax_params.get((yparam, 'y'), None)

            ax = self.create_axis(view, loc, row, col, xax=xax, yax=yax)
            loc_axes = self.axes.setdefault(loc, {})
            loc_axes[row, col] = ax
            if xparam is not None and xax is None:
                self.ax_params[xparam, 'x'] = ax
            if yparam is not None and yax is None:
                self.ax_params[yparam, 'y'] = ax
            return ax

    def create_axis(self, view, loc, row, col, xax=None, yax=None):
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
        ax = fig.add_subplot(subspec, sharex=xax, sharey=yax)
        try:
            xlabel, xunit, kwargs = self.labels[view, row, col, 'x']
            if xunit:
                xlabel += ' / {}'.format(xunit)
            ax.set_xlabel(xlabel, **kwargs)
        except KeyError:
            pass
        try:
            xmin, xmax = self.xlim[view, row, col]
            if np.nan not in (xmin, xmax):
                offs = 0.025 * abs(xmax - xmin)
                ax.set_xlim(xmin - offs, xmax + offs)
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

    def open_window(self, window=0):
        fig = self.windows.get(window, None)
        if fig:
            return fig
        nb_backends = 'inline', 'notebook', 'nbAgg', 'ipympl'
        backend = plt.matplotlib.get_backend()
        if any(nbb in backend for nbb in nb_backends):
            fig = plt.figure()
            label = self.get_window_label(window)
            fig.canvas.set_window_title(label)
            fig.set_label(label)
        else:
            label = self.get_window_label(window)
            fig = plt.figure(label)
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
    def _path_to_label(path, task, format='short'):
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
                if format == 'short':
                    s = r'${}_\mathsf{{{}}}$'
                    s = s.format(name, param._task.name)
                else:
                    s = r'${}$'.format(repr(param).strip('<>'))
            label.append(s)
        return ' | '.join(label)

    @staticmethod
    def _path_to_short_label(path, task):
        if not path:
            return ''
        label = []
        for arg in path:
            if arg._task is not task:
                label.append(arg._task.name)
        name = arg.name.replace('_', '\_')
        s = r'${}$'.format(name)
        label.append(s)
        return ' | '.join(label)

    @staticmethod
    def _value_str(value):
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float, complex)):
            return '{:.7g}'.format(value)
        elif isinstance(value, Iterable):
            try:
                return r'$[{:.4g}, \ldots]$'.format(value[0])
            except TypeError:
                value = repr(value)
                if len(value) > 8:
                    return value[:8] + '...'
                else:
                    return value
        else:
            return repr(value)

    def on_pick(self, event):
        idx = event.ind[0]
        line = event.artist
        lmngr = self.lines[line]
        line.figure.canvas.flush_events()

        xval, yval, cidxs = lmngr.set_cursor_selection(line, idx)
        log.debug('SELCETION: idx = {}'.format(idx))
        log.debug('    xval: {}'.format(xval))
        log.debug('    yval: {}'.format(yval))
        log.debug('   cidxs: {}'.format(cidxs))
        # ypath legend
        leg_lines = []
        leg_labels = []
        _params = set()
        ylabel = self._path_to_short_label(lmngr.ypath, lmngr.task)
        xlabel = self._path_to_short_label(lmngr.xpath, lmngr.task)
        yval_str = self._value_str(yval)
        if xlabel and lmngr.xpath[-1].name in lmngr.task.returns:
            label = '{}({}) = {}'.format(ylabel, xlabel, yval_str)
        else:
            label = '{} = {}'.format(ylabel, yval_str)
        _params.add(tuple(lmngr.ypath))
        # xpath legend
        if lmngr.xpath:
            label += '\n{} = {}'.format(xlabel, self._value_str(xval))
            _params.add(tuple(lmngr.xpath))
        else:
            label += '\n' 'index = {}'.format(idx)
        leg_labels.append(label)
        leg_lines.append(line)
        # args legend
        sq_labels = []
        # add squeezed args
        for path in lmngr.squeeze:
            if not path or path in _params or len(path[-1]._states) <= 1:
                continue
            name = self._path_to_short_label(path, lmngr.task)
            _cidx = lmngr.task._get_cidx_from_path(path, cidxs[0], fold=0)
            if path in lmngr.squeeze:
                val = path[-1].get_cache(_cidx)
            else:
                val = path[-1].get_cache(_cidx[idx])
            val_str = self._value_str(val)
            label = '{} = {}'.format(name, val_str)
            sq_labels.append(label)
            _params.add(tuple(path))
        idx_invis = []
        if sq_labels:
            leg_lines.append(line)
            leg_labels.append('\n'.join(sq_labels))
            idx_invis.append(len(leg_labels) - 1)
        # add line args
        arg_labels = []
        arg_states = lmngr.get_key(cidxs[0])
        for state, path in zip(arg_states, lmngr._key_paths):
            if tuple(path) in _params or len(path[-1]._states) <= 1:
                continue
            name = self._path_to_short_label(path, lmngr.task)
            if state is None:
                val_str = '_squeezed_'
            else:
                val = path[-1].get_value(state)
                val_str = self._value_str(val)
            label = '{} = {}'.format(name, val_str)
            arg_labels.append(label)
            _params.add(tuple(path))
        if arg_labels:
            leg_lines.append(line)
            leg_labels.append('\n'.join(arg_labels))
            idx_invis.append(len(leg_labels) - 1)
        # create legend
        leg = lmngr.ax.legend(leg_lines, leg_labels, loc=self.legend_loc)
        for _idx in idx_invis:
            ln = leg.legendHandles[_idx]
            ln.set_visible(False)
            ln._legmarker.set_visible(False)
        view = self.views[lmngr.task]
        if len(self.tasks[view]) > 1:
            leg.set_title(lmngr.task.name)

        legs = {}
        for task, confs in self.lms[lmngr.loc].items():
            if task is lmngr.task:
                for lm in confs.values():
                    if lm is lmngr:
                        continue
                    if lmngr.xpath and lmngr.xpath == lm.xpath:
                        ln_idx = idx
                    else:
                        ln_idx = None
                    lines = lm.set_selection(cidxs, ln_idx)

                    lms = legs.setdefault(lm.ax, {})
                    lms.setdefault(lm, {}).update(lines)
            else:
                for lm in confs.values():
                    if lm._cursor is not None:
                        lm._cursor.set_visible(False)
                    for cursor in lm._selections:
                        cursor.set_visible(False)

        for ax in self.axes[lmngr.loc].values():
            if ax is line.axes:
                continue
            if ax not in legs:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
                continue
            leg_lines = []
            leg_labels = []
            for lm, lines in legs[ax].items():
                ylabel = self._path_to_short_label(lm.ypath, lm.task)
                xlabel = self._path_to_short_label(lm.xpath, lm.task)
                if xlabel and lm.xpath[-1].name in lm.task.returns:
                    label = '{}({})'.format(ylabel, xlabel)
                else:
                    label = ylabel
                for num, (ln, xydata) in enumerate(lines.items()):
                    leg_lines.append(ln)
                    leg_labels.append('')
                leg_labels[-1] = label
                if num < 1 and len(xydata) < 2:
                    xdata, ydata = xydata[0]
                    #~ leg_labels[-1] += ' = {:.7g}'.format(ydata)
                    if ydata is not None:
                        ydata_str = self._value_str(ydata)
                        leg_labels[-1] += ' = {}'.format(ydata_str)
                        if not lm.xpath:
                            xlabel = 'index'
                        xdata_str = self._value_str(xdata)
                        leg_labels[-1] += '\n{} = {}'.format(xlabel,
                                                             xdata_str)

            leg = ax.legend(leg_lines, leg_labels, loc=self.legend_loc)
            view = self.views[lmngr.task]
            if len(self.tasks[view]) > 1:
                leg.set_title(lm.task.name)

        fig = line.get_figure()
        fig.canvas.draw()
        #~ fig.canvas.flush_events()

        self.selection = dict(task=lmngr.task,
                              cidxs=cidxs,
                              _lm=lmngr,
                              _line=line,
                              _idx=idx)


class LineManager:
    def __init__(self, loc, ax, task, pm, xpath, ypath,
                 squeeze, accumulate, use_cursor, xsort,
                 kwargs):
        self.loc = loc
        self.ax = ax
        self.task = task

        self.use_cursor = use_cursor
        self.kwargs = kwargs
        self.pm = pm
        self.xpath = xpath
        self.ypath = ypath
        self.xsort = xsort

        if squeeze == '*':
            squeeze = task.args._get_arg_paths()
        self._tasksweep = taskmanager._PlotSweep(task, squeeze)
        self._datas = {}

        # get mask from accumulate-paths
        acc_paths = self._key_paths if accumulate == '*' else accumulate
        self.acc_paths = []
        for path in acc_paths:
            zipped = []
            for param in path:
                task = param._task
                try:
                    level = task.args._nested_args[param]
                    zipped.append(task.args._nested_levels[level])
                except KeyError:
                    zipped.append([param])
            self.acc_paths += [list(params) for params in product(*zipped)]
        self.mask = [path in self.acc_paths for path in self._key_paths]

        self.lines = {}     # key: line
        self.cidxs = {}     # line: [cidx, ...]
        self.clen = 0

        self.cursor = None
        self._selections = []
        self._cursor = None

        # used for caching
        self._keys = {}

    @property
    def _key_paths(self):
        return self._tasksweep._key_paths

    @property
    def squeeze(self):
        return [tuple(path) for path in self._tasksweep.squeeze]

    def create_cursor(self):
        cursor, = self.ax.plot([], [],
            linewidth=7,
            marker='o',
            markersize=14,
            alpha=0.5,
            zorder=1,
        )
        return cursor

    def set_cursor_selection(self, line, idx=-1):
        xval = line.get_xdata()[idx]
        yval = line.get_ydata()[idx]
        if self._cursor is None:
            self._cursor = self.create_cursor()
        cursor = self._cursor
        cursor.set_data([xval, yval])
        cursor.set_marker('o')
        color = line.get_color()
        cursor.set_color(color)  # or: shade_color(color, 50)
        cursor.set_visible(True)

        cidxs = self.cidxs[line]
        if self.squeeze:
            cidxs = [cidxs[idx]]
        self.show_accumulate_lines(cidxs[0])
        for cursor in self._selections:
            cursor.set_visible(False)
        return xval, yval, cidxs

    def set_selection(self, cidxs, ln_idx=None):
        lines = {}
        self.show_accumulate_lines(cidxs)
        for num, cidx in enumerate(cidxs):
            key = self.get_key(cidx)
            line = self.lines[key]
            self._show_line(line, True)
            if (not self.squeeze
                and not any(self.mask)
                and ln_idx is None
            ):
                lines.setdefault(line, []).append((None, None))
                num = num - 1
                continue
            try:
                cursor = self._selections[num]
            except IndexError:
                cursor = self.create_cursor()
                self._selections.append(cursor)
            line_xdata = line.get_xdata()
            line_ydata = line.get_ydata()
            if self.squeeze:
                line_cidxs = self.cidxs[line]
                idx = line_cidxs.index(cidx)
                xdata = line_xdata[idx]
                ydata = line_ydata[idx]
                cursor.set_marker('o')
                cursor.set_linestyle('')
            elif ln_idx is not None and ln_idx < len(line_xdata):
                xdata = line_xdata[ln_idx]
                ydata = line_ydata[ln_idx]
                cursor.set_marker('o')
                cursor.set_linestyle('')
            elif len(self.lines) == 1:
                num = num - 1
                lines.setdefault(line, []).append((None, None))
                continue
            else:
                xdata = line_xdata
                ydata = line_ydata
                cursor.set_marker(None)
                cursor.set_linestyle('-')
            cursor.set_data([xdata, ydata])
            color = line.get_color()
            cursor.set_color(color)  # or: shade_color(color, 50)
            cursor.set_visible(True)
            lines.setdefault(line, []).append((xdata, ydata))
        for cursor in self._selections[num+1:]:
            cursor.set_visible(False)
        if self._cursor:
            self._cursor.set_visible(False)
        return lines

    def get_key(self, cidx):
        """like tasksweep.get_key() but append None for mask feature"""
        if cidx in self._keys:
            return self._keys[cidx]
        states = []
        for path in self._key_paths:
            if (not self.squeeze
                and path[0] is not self.ypath[0]
                and self.ypath[0] in self.task.args
            ):
                states.append(None)
                continue
            _cidx = self.task._get_cidx_from_path(path, cidx)
            state = path[-1].get_arg_state(_cidx)
            states.append(state)
        states = tuple(states)
        self._keys[cidx] = states
        return states

    def update(self):
        ydata = []
        newlines = []

        tasksweep = self._tasksweep
        tasksweep.configure()
        for state in tasksweep:
            _, __, cidx = state
            #~ key = tasksweep._keys[cidx]
            key = self.get_key(cidx)
            cidxs = tasksweep.get_cidxs(*state)

            if key not in self.lines:
                line = self.newline()
                self.lines[key] = line
                self.cidxs[line] = cidxs
                newlines.append(line)
                self.pm.lines[line] = self
            else:
                line = self.lines[key]
                self.cidxs[line].extend(cidxs)
                if not self.squeeze:
                    continue

            yvals = self.task.get_value_from_path(self.ypath, cidxs)
            ydata = line.get_ydata()
            ydata = np.append(ydata, yvals)

            if self.xpath:
                xvals = self.task.get_value_from_path(self.xpath, cidxs)
                xdata = line.get_xdata()
                xdata = np.append(xdata, xvals)
                if self.xsort:
                    xidxs = np.argsort(xdata)
                    xdata = xdata[xidxs]
                    ydata = ydata[xidxs]
                    self.cidxs[line] = [self.cidxs[line][n] for n in xidxs]
            else:
                xdata = np.arange(len(ydata))

            line.set_ydata(ydata)
            line.set_xdata(xdata)

        if len(ydata):
            self.set_cursor_selection(line)

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

    def show_accumulate_lines(self, cidxs):
        if not isinstance(cidxs, (list, tuple)):
            cidxs = [cidxs]
        states = []
        for cidx in cidxs:
            states.append(self.get_key(cidx))
        for key, line in self.lines.items():
            value = 0
            for state in states:
                value |= all(self.compare(key, state))
            self._show_line(line, value)

    @staticmethod
    def _show_line(line, value):
        line.set_visible(value)
        line.set_picker(10 if value else None)

    def compare(self, states, other):
        for s1, s2, m in zip(states, other, self.mask):
            if None in (s1, s2):
                yield True
            else:
                yield m or s1 == s2

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
