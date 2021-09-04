from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

from plagdef.model import models
from plagdef.model.models import MatchType
from plagdef.util import truncate


@dataclass(frozen=True)
class DocumentPairMatches:
    doc1: models.Document
    doc2: models.Document
    matches: list[models.Match]

    @classmethod
    def from_model(cls, model: models.DocumentPairMatches, match_type) -> DocumentPairMatches:
        typed_matches = model.list(match_type)
        if len(typed_matches):
            d1 = model.doc1 if model.doc1.name < model.doc2.name else model.doc2
            d2 = model.doc2 if d1 == model.doc1 else model.doc1
            return DocumentPairMatches(d1, d2,
                                       sorted(typed_matches,
                                              key=lambda m, doc1=d1: m.frag_from_doc(doc1).start_char))

    def __len__(self):
        return len(self.matches)


class ResultsTableModel(QAbstractTableModel):
    def __init__(self, match_type: MatchType, doc_pair_matches: list[models.DocumentPairMatches]):
        super().__init__()
        self._doc_pair_matches = [DocumentPairMatches.from_model(matches, match_type) for matches in doc_pair_matches]
        self._doc_pair_matches = sorted(filter(None, self._doc_pair_matches), key=lambda m: m.doc1.name)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        headers = ['Document 1', 'Document 2']
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return headers[section]

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._doc_pair_matches)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        doc_pair = self._doc_pair_matches[index.row()].doc1, self._doc_pair_matches[index.row()].doc2
        if role == Qt.DisplayRole:
            name = doc_pair[index.column()].name
            return truncate(name, 50)
        elif role == Qt.ForegroundRole:
            return QColor(255, 255, 255)
        elif role == Qt.ToolTipRole:
            return doc_pair[index.column()].path

    def doc_pair_matches(self, index: QModelIndex) -> DocumentPairMatches:
        return self._doc_pair_matches[index.row()]
