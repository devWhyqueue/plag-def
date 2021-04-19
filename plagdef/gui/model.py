from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

from plagdef.model import models


@dataclass(frozen=True)
class ResultRow:
    doc1: str
    doc2: str


@dataclass(frozen=True)
class DocumentPairMatches:
    doc1: models.Document
    doc2: models.Document
    plag_type: models.PlagiarismType
    matches: list[models.Match]

    def __len__(self):
        return len(self.matches)


class ResultsTableModel(QAbstractTableModel):
    def __init__(self, doc_pair_matches: list[models.DocumentPairMatches]):
        super().__init__()
        self._rows = []
        self._doc_pair_matches = []
        for matches in sorted(doc_pair_matches, key=lambda m: m.plag_type):
            doc1, doc2 = matches.doc_pair
            doc1_path = doc1.path if len(doc1.path) < 50 else f'...{doc1.path[len(doc1.path) - 50:]}'
            doc2_path = doc2.path if len(doc2.path) < 50 else f'...{doc2.path[len(doc2.path) - 50:]}'
            self._rows.append(ResultRow(doc1_path, doc2_path))
            self._doc_pair_matches.append(
                DocumentPairMatches(doc1, doc2, matches.plag_type,
                                    sorted(matches.list(), key=lambda m: m.frag_from_doc(doc1).start_char)))

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        headers = ['Document 1', 'Document 2']
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return headers[section]

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._rows)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        doc_pair = self._rows[index.row()]
        doc_pair_attributes = [doc_pair.doc1, doc_pair.doc2]
        if role == Qt.DisplayRole:
            return doc_pair_attributes[index.column()]
        elif role == Qt.ForegroundRole:
            plag_type = self._doc_pair_matches[index.row()].plag_type
            if plag_type == models.PlagiarismType.VERBATIM:
                return QColor(242, 141, 1)
            elif plag_type == models.PlagiarismType.INTELLIGENT:
                return QColor(242, 191, 121)
            elif plag_type == models.PlagiarismType.SUMMARY:
                return QColor(255, 255, 255)

    def doc_pair_matches(self, index: QModelIndex) -> DocumentPairMatches:
        return self._doc_pair_matches[index.row()]
