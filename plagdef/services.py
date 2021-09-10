from __future__ import annotations

import logging
from pathlib import Path

from click import UsageError

from plagdef.config import settings
from plagdef.model.detection import DocumentMatcher
from plagdef.model.download import download_all_external_sources
from plagdef.model.models import DocumentPairMatches, Document
from plagdef.model.pipeline.preprocessing import UnsupportedLanguageError
from plagdef.repositories import UnsupportedFileFormatError, DocumentPickleRepository, DocumentFileRepository, \
    FileRepository

log = logging.getLogger(__name__)


def find_matches(doc_repo, archive_repo=None, common_doc_repo=None, config=settings) \
    -> list[DocumentPairMatches]:
    try:
        doc_matcher = DocumentMatcher(config)
        archive_docs = None
        if archive_repo:
            archive_docs = _preprocess_docs(doc_matcher, config['lang'], config['ser'], archive_repo, common_doc_repo)
        docs = _preprocess_docs(doc_matcher, config['lang'], config['ser'], doc_repo, common_doc_repo)
        if config['download_path']:
            _save_all_external_sources(docs, config['download_path'])
            external_docs = _preprocess_docs(doc_matcher, config['lang'], config['ser'],
                                             DocumentFileRepository(Path(config['download_path'])), common_doc_repo)
            archive_docs = archive_docs.union(external_docs) if archive_docs else external_docs
        doc_pair_matches = doc_matcher.find_matches(docs, archive_docs)
        return doc_pair_matches
    except (UnsupportedFileFormatError, UnsupportedLanguageError) as e:
        raise UsageError(str(e)) from e


def _preprocess_docs(doc_matcher, lang, use_serialization, doc_repo, common_doc_repo=None) -> set[Document]:
    common_dir_path = common_doc_repo.base_path if common_doc_repo else None
    common_docs = common_doc_repo.list() if common_doc_repo else None
    docs = doc_repo.list()
    if use_serialization:
        doc_ser = DocumentPickleRepository(doc_repo.base_path, common_dir_path)
        prep_docs = {d for d in doc_ser.list() if d in docs}
        unprep_docs = docs.difference(prep_docs)
        doc_matcher.preprocess(lang, unprep_docs, common_docs)
        preprocessed_docs = prep_docs.union(unprep_docs)
        doc_ser.save(preprocessed_docs)
    else:
        doc_matcher.preprocess(lang, docs, common_docs)
        preprocessed_docs = docs
    return preprocessed_docs


def _save_all_external_sources(docs, download_path):
    external_sources = download_all_external_sources(docs, Path(download_path))
    external_sources_repo = FileRepository(Path(download_path))
    external_sources_repo.save_all(external_sources)


def write_json_reports(matches: list[DocumentPairMatches], repo):
    [repo.save(m) for m in matches]
