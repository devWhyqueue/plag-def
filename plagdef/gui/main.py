from __future__ import annotations

import os
import signal
import sys
import traceback

from PySide6.QtCore import QRunnable, Slot, QObject, Signal, QThreadPool
from PySide6.QtWidgets import QApplication

# noinspection PyUnresolvedReferences
import plagdef.gui.resources
from plagdef.app import find_matches, reanalyze_pair
from plagdef.gui.controllers import HomeController, LoadingController, ErrorController, NoResultsController, \
    ResultController
from plagdef.gui.views import MainWindow

app = None


class MyQtApp(QApplication):
    def __init__(self):
        super().__init__()
        self.aboutToQuit.connect(lambda: os.kill(os.getpid(), signal.SIGINT))
        self._ctrls = [HomeController(), LoadingController(), ErrorController(), NoResultsController(),
                       ResultController()]
        self._views = [ctrl.view for ctrl in self._ctrls]
        self.window = MainWindow(self._views)
        global app
        app = self

    def find_matches(self, docdir: tuple[str, bool], archive_docdir: [str, bool],
                     common_docdir: [str, bool], on_success, on_error):
        worker = Worker(find_matches, docdir, archive_docdir, common_docdir)
        worker.signals.result.connect(on_success)
        worker.signals.error.connect(on_error)
        pool = QThreadPool.globalInstance()
        pool.start(worker)

    def reanalyze_pair(self, doc1, doc2, sim: float, on_success, on_error):
        worker = Worker(reanalyze_pair, doc1, doc2, sim)
        worker.signals.result.connect(on_success)
        worker.signals.error.connect(on_error)
        pool = QThreadPool.globalInstance()
        pool.start(worker)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)


class WorkerSignals(QObject):
    error = Signal(tuple)
    result = Signal(object)
