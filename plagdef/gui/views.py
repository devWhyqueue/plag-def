from __future__ import annotations

import sys
from pathlib import Path

import pkg_resources
from PySide6.QtCore import QFile, QIODevice, Qt
from PySide6.QtGui import QCursor, QMovie
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QButtonGroup, QMainWindow, QFileDialog, QDialog

from plagdef.config import settings
from plagdef.gui.model import ResultsTableModel, DocumentPairMatches
from plagdef.model import models
from plagdef.util import version, truncate

UI_FILES = {
    'main_window': pkg_resources.resource_filename(__name__, 'ui/main_window.ui'),
    'home_widget': pkg_resources.resource_filename(__name__, 'ui/home_widget.ui'),
    'loading_widget': pkg_resources.resource_filename(__name__, 'ui/loading_widget.ui'),
    'error_widget': pkg_resources.resource_filename(__name__, 'ui/error_widget.ui'),
    'no_results_widget': pkg_resources.resource_filename(__name__, 'ui/no_results_widget.ui'),
    'results_widget': pkg_resources.resource_filename(__name__, 'ui/results_widget.ui'),
    'matches_dialog': pkg_resources.resource_filename(__name__, 'ui/matches_dialog.ui'),
    'msg_dialog': pkg_resources.resource_filename(__name__, 'ui/msg_dialog.ui'),
    'settings_dialog': pkg_resources.resource_filename(__name__, 'ui/settings_dialog.ui'),
}


class MainWindow:
    def __init__(self, views: list[View]):
        self._window = _load_ui_file(Path(UI_FILES['main_window']))
        self._views = views
        self._configure()

    def _configure(self):
        self._window.setWindowTitle(f'PlagDef v{version()}')
        for view in self._views:
            self._window.stacked_widget.addWidget(view.widget)

    def switch_to(self, view_cls: type, data=None):
        idx = self._window.stacked_widget.currentIndex()
        for view in self._views:
            if type(view) == view_cls:
                self._views[idx].on_destroy()
                self._window.stacked_widget.setCurrentWidget(view.widget)
                view.on_init(data)

    def show(self):
        self._window.show()


class View:
    def on_init(self, data=None):
        pass

    def on_destroy(self):
        pass


class HomeView(View):
    def __init__(self, lang=settings['lang']):
        self.widget = _load_ui_file(Path(UI_FILES['home_widget']))
        self._configure(lang)

    def _configure(self, lang):
        self.widget.lang_button_group = QButtonGroup()
        self.widget.lang_button_group.addButton(self.widget.ger_button)
        self.widget.lang_button_group.addButton(self.widget.eng_button)
        self.widget.ger_button.click() if lang == 'ger' else self.widget.eng_button.click()
        [element.setVisible(False)
         for element in (self.widget.archive_dir_label, self.widget.archive_rmdir_button, self.widget.docs_dir_label,
                         self.widget.docs_rmdir_button, self.widget.common_dir_label, self.widget.common_rmdir_button)]
        [button.setCursor(QCursor(Qt.PointingHandCursor))
         for button in (self.widget.ger_button, self.widget.eng_button, self.widget.archive_dir_button,
                        self.widget.docs_dir_button, self.widget.common_dir_button, self.widget.detect_button,
                        self.widget.archive_rmdir_button, self.widget.docs_rmdir_button,
                        self.widget.common_rmdir_button, self.widget.open_button, self.widget.settings_button)]

    @property
    def archive_rec(self):
        return self.widget.archive_rec_check_box.isChecked()

    @property
    def docs_rec(self):
        return self.widget.docs_rec_check_box.isChecked()

    @property
    def common_rec(self):
        return self.widget.common_rec_check_box.isChecked()

    @property
    def lang(self):
        return 'ger' if self.widget.lang_button_group.checkedButton() == self.widget.ger_button else 'eng'

    def register_for_signals(self, select_lang=None, open_report_dir=None, select_archive_dir=None, rm_archive_dir=None,
                             select_docs_dir=None, rm_docs_dir=None, select_common_dir=None, rm_common_dir=None,
                             detect=None, settings=None):
        self.widget.lang_button_group.buttonClicked.connect(lambda: select_lang())
        self.widget.settings_button.clicked.connect(lambda: settings())
        self.widget.open_button.clicked.connect(lambda: open_report_dir())
        self.widget.archive_dir_button.clicked.connect(lambda: select_archive_dir())
        self.widget.archive_rmdir_button.clicked.connect(lambda: rm_archive_dir())
        self.widget.docs_dir_button.clicked.connect(lambda: select_docs_dir())
        self.widget.docs_rmdir_button.clicked.connect(lambda: rm_docs_dir())
        self.widget.common_dir_button.clicked.connect(lambda: select_common_dir())
        self.widget.common_rmdir_button.clicked.connect(lambda: rm_common_dir())
        self.widget.detect_button.clicked.connect(lambda: detect())

    def archive_dir_selected(self, folder_name: str):
        self._dir_selected(folder_name, self.widget.archive_dir_button, self.widget.archive_rmdir_button,
                           self.widget.archive_dir_label, 12)

    def archive_dir_removed(self):
        self._dir_removed(self.widget.archive_dir_button, self.widget.archive_rmdir_button,
                          self.widget.archive_dir_label)

    def docs_dir_selected(self, folder_name: str):
        self._dir_selected(folder_name, self.widget.docs_dir_button, self.widget.docs_rmdir_button,
                           self.widget.docs_dir_label, 16)
        self.widget.detect_button.setEnabled(True)

    def docs_dir_removed(self):
        self._dir_removed(self.widget.docs_dir_button, self.widget.docs_rmdir_button, self.widget.docs_dir_label)
        self.widget.detect_button.setEnabled(False)

    def common_dir_selected(self, folder_name: str):
        self._dir_selected(folder_name, self.widget.common_dir_button, self.widget.common_rmdir_button,
                           self.widget.common_dir_label, 12)

    def common_dir_removed(self):
        self._dir_removed(self.widget.common_dir_button, self.widget.common_rmdir_button, self.widget.common_dir_label)

    def _dir_selected(self, folder_name: str, dir_button, rm_button, label, font_size: int):
        dir_button.setEnabled(False)
        label.setText(
            f'<html><head/><body><p align="center"><span style="font-size:{font_size}pt; color:#ffffff;"> '
            f'{folder_name}</span></p></body></html>')
        label.setVisible(True)
        rm_button.setVisible(True)

    def _dir_removed(self, button, rm_button, label):
        label.setVisible(False)
        rm_button.setVisible(False)
        button.setEnabled(True)

    def on_destroy(self):
        self.archive_dir_removed()
        self.docs_dir_removed()
        self.common_dir_removed()


class FileDialog(QFileDialog):
    def __init__(self):
        super().__init__()
        self.setFileMode(QFileDialog.Directory)
        self.selected_dir = None

    def open(self) -> bool:
        if self.exec_() == QDialog.Accepted:
            self.selected_dir = self.selectedFiles()[0]
            return True


class LoadingView(View):
    def __init__(self):
        self._movie = QMovie(':/loading.gif')
        self.widget = _load_ui_file(Path(UI_FILES['loading_widget']))
        self._configure()

    def _configure(self):
        self.widget.loading_movie_label.setMovie(self._movie)

    def on_init(self, data=None):
        self._movie.start()

    def on_destroy(self):
        self._movie.stop()


class NoResultsView(View):
    def __init__(self):
        self.widget = _load_ui_file(Path(UI_FILES['no_results_widget']))
        self._configure()

    def _configure(self):
        self.widget.again_button_no_res.setCursor(QCursor(Qt.PointingHandCursor))

    def register_for_signals(self, again=None):
        self.widget.again_button_no_res.clicked.connect(lambda: again())


class ErrorView(View):
    def __init__(self):
        self.widget = _load_ui_file(Path(UI_FILES['error_widget']))
        self._configure()

    def _configure(self):
        self.widget.again_button_err.setCursor(QCursor(Qt.PointingHandCursor))

    def on_init(self, data=None):
        self.widget.error_msg_label.setText(
            f'<html><head/><body><p align="center"><span style=" font-size:12pt; color:#ffffff;"> '
            f'{data}</span></p></body></html>')
        self.widget.error_msg_label.setWordWrap(True)

    def register_for_signals(self, again=None):
        self.widget.again_button_err.clicked.connect(lambda: again())


class ResultView(View):
    def __init__(self):
        self.widget = _load_ui_file(Path(UI_FILES['results_widget']))
        self._configure()
        self.doc_pair_matches = None

    def _configure(self):
        self.widget.export_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.widget.again_button_res.setCursor(QCursor(Qt.PointingHandCursor))

    def on_init(self, data=None):
        self.doc_pair_matches = data
        self.widget.doc_pairs_label.setText(f"Found {len(data) if len(data) else 'no'} suspicious document pair"
                                            f"{'s' if len(data) > 1 else ''}.")
        self._set_table_model(self.widget.verbatim_results, models.MatchType.VERBATIM, data)
        self._set_table_model(self.widget.intelligent_results, models.MatchType.INTELLIGENT, data)
        self._set_table_model(self.widget.summary_results, models.MatchType.SUMMARY, data)
        self._hide_empty_tables()

    def _set_table_model(self, table, match_type, pairs):
        table.setModel(ResultsTableModel(match_type, pairs))
        table.resizeRowsToContents()
        table.resizeColumnsToContents()
        table.parentWidget().adjustSize()

    def _hide_empty_tables(self):
        self.widget.doc_pair_tabs.setTabVisible(0, bool(self.widget.verbatim_results.model().rowCount()))
        self.widget.doc_pair_tabs.setTabVisible(1, bool(self.widget.intelligent_results.model().rowCount()))
        self.widget.doc_pair_tabs.setTabVisible(2, bool(self.widget.summary_results.model().rowCount()))

    @property
    def selected_matches(self) -> list[models.DocumentPairMatches]:
        matches = []
        verbatim_indices = self.widget.verbatim_results.selectionModel().selectedIndexes()
        verbatim_indices = filter(lambda i: i.column() == 0, verbatim_indices)
        for index in verbatim_indices:
            verbatim_matches = self.widget.verbatim_results.model().doc_pair_matches(index)
            matches.extend(verbatim_matches.matches)
        intelligent_indices = self.widget.intelligent_results.selectionModel().selectedIndexes()
        intelligent_indices = filter(lambda i: i.column() == 0, intelligent_indices)
        for index in intelligent_indices:
            intelligent_matches = self.widget.intelligent_results.model().doc_pair_matches(index)
            matches.extend(intelligent_matches.matches)
        summary_indices = self.widget.summary_results.selectionModel().selectedIndexes()
        summary_indices = filter(lambda i: i.column() == 0, summary_indices)
        for index in summary_indices:
            summary_matches = self.widget.summary_results.model().doc_pair_matches(index)
            matches.extend(summary_matches.matches)
        return models.DocumentPairMatches.from_matches(matches)

    def register_for_signals(self, export=None, again=None, select_pair=None):
        self.widget.export_button.clicked.connect(lambda: export())
        self.widget.again_button_res.clicked.connect(lambda: again())
        self.widget.verbatim_results.doubleClicked.connect(
            lambda index: select_pair(index.model().doc_pair_matches(index)))
        self.widget.intelligent_results.doubleClicked.connect(
            lambda index: select_pair(index.model().doc_pair_matches(index)))
        self.widget.summary_results.doubleClicked.connect(
            lambda index: select_pair(index.model().doc_pair_matches(index)))


class MatchesDialog:
    def __init__(self):
        self.widget = _load_ui_file(Path(UI_FILES['matches_dialog']))
        self._configure()
        self._doc_pair_matches = None
        self._selected = 0

    def _configure(self):
        self.widget.doc1_label.setCursor(QCursor(Qt.PointingHandCursor))
        self.widget.doc2_label.setCursor(QCursor(Qt.PointingHandCursor))
        self.widget.reanalyze_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.widget.sim_slider.valueChanged.connect(lambda val: self._update_label(val))

    def _update_label(self, sim):
        self.widget.value_label.setText(str(sim / 10))

    @property
    def doc1(self):
        return self._doc_pair_matches.doc1

    @property
    def doc2(self):
        return self._doc_pair_matches.doc2

    @property
    def sim_threshold(self) -> float:
        return float(self.widget.value_label.text())

    @property
    def match_type(self):
        return self._doc_pair_matches.matches[0].type

    def open(self, doc_pair_matches: DocumentPairMatches):
        self.widget.value_label.setText(str(settings['min_cos_sim']))
        self.widget.sim_slider.setValue(settings['min_cos_sim'] * 10)
        self.set_data(doc_pair_matches)
        self.widget.exec_()

    def set_data(self, doc_pair_matches):
        self._doc_pair_matches = doc_pair_matches
        self.widget.doc1_label.setText(truncate(self._doc_pair_matches.doc1.name, 50))
        self.widget.doc1_path.setText(f'({truncate(self._doc_pair_matches.doc1.path, 65)})')
        self.widget.doc2_label.setText(truncate(self._doc_pair_matches.doc2.name, 50))
        self.widget.doc2_path.setText(f'({truncate(self._doc_pair_matches.doc2.path, 65)})')
        self._selected = 0
        self._show_match()

    def _show_match(self):
        doc1, doc2 = self._doc_pair_matches.doc1, self._doc_pair_matches.doc2
        self.widget.doc1_text.setText(self._doc_pair_matches.matches[self._selected].frag_from_doc(doc1).text)
        self.widget.doc2_text.setText(self._doc_pair_matches.matches[self._selected].frag_from_doc(doc2).text)
        self.widget.page_label.setText(f'{self._selected + 1}/{len(self._doc_pair_matches)}')
        self.widget.prev_button.setVisible(True)
        self.widget.next_button.setVisible(True)
        if self._selected == 0:
            self.widget.prev_button.setVisible(False)
        if self._selected + 1 == len(self._doc_pair_matches):
            self.widget.next_button.setVisible(False)

    def register_for_signals(self, prev_match=None, next_match=None, open_doc=None, reanalyze=None):
        self.widget.prev_button.clicked.connect(lambda: prev_match())
        self.widget.next_button.clicked.connect(lambda: next_match())
        self.widget.doc1_label.clicked.connect(lambda: open_doc(self._doc_pair_matches.doc1.path))
        self.widget.doc2_label.clicked.connect(lambda: open_doc(self._doc_pair_matches.doc2.path))
        self.widget.reanalyze_button.clicked.connect(lambda: reanalyze())

    def prev_match(self):
        self._selected -= 1
        self._show_match()

    def next_match(self):
        self._selected += 1
        self._show_match()

    def reanalyzing(self, loading: bool):
        self.widget.sim_slider.setEnabled(not loading)
        self.widget.reanalyze_button.setEnabled(not loading)


class MessageDialog:
    def __init__(self, text):
        self.widget = _load_ui_file(Path(UI_FILES['msg_dialog']))
        self.widget.msg_label.setText(text)
        self.widget.exec_()


class SettingsDialog:
    def __init__(self):
        self.widget = _load_ui_file(Path(UI_FILES['settings_dialog']))
        self.widget.sim_slider.valueChanged.connect(lambda val: self._update_label(val))

    def _update_label(self, sim):
        self.widget.value_label.setText(str(sim / 10))

    @property
    def similarity_threshold(self) -> float:
        return float(self.widget.value_label.text())

    @property
    def ocr(self) -> bool:
        return self.widget.ocr_check_box.isChecked()

    @property
    def download_path(self) -> bool:
        return self.widget.es_line_edit.text()

    def register_for_signals(self, select_download_path=None):
        self.widget.es_button.clicked.connect(lambda: select_download_path())

    def open(self, ocr=None, sim=None, download_path=None):
        ocr = settings['ocr'] if not ocr else ocr
        sim = settings['min_cos_sim'] if not sim else sim
        download_path = settings['download_path'] if not download_path else download_path
        self.widget.ocr_check_box.setChecked(ocr)
        self.widget.es_line_edit.setText(download_path)
        self.widget.value_label.setText(str(sim))
        self.widget.sim_slider.setValue(sim * 10)
        self.widget.exec_()

    def download_dir_selected(self, dir_path):
        self.widget.es_button.setText("Open..." if not dir_path else "Clear")
        self.widget.es_line_edit.setText(dir_path)


def _load_ui_file(path: Path) -> QMainWindow:
    main_ui = QFile(str(path))
    if not main_ui.open(QIODevice.ReadOnly):
        print(f"Cannot open {path}: {main_ui.errorString()}")
        sys.exit(-1)
    loader = QUiLoader()
    window = loader.load(main_ui)
    main_ui.close()
    if not window:
        print(loader.errorString())
        sys.exit(-1)
    return window
