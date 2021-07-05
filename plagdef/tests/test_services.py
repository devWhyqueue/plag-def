from unittest.mock import patch

import pytest
from click import UsageError

from plagdef.model.detection import DocumentMatcher
from plagdef.model.models import DocumentPairMatches, Match, Fragment, MatchType
from plagdef.model.pipeline.preprocessing import Document, UnsupportedLanguageError
from plagdef.repositories import UnsupportedFileFormatError, DocumentPairMatchesJsonRepository, DocumentPickleRepository
from plagdef.services import find_matches, write_json_reports, _preprocess_docs
from plagdef.tests.fakes import DocumentFakeRepository, FakeDocumentMatcher


def test_preprocess_docs(tmp_path):
    doc_matcher = FakeDocumentMatcher()
    docs = {Document('doc1', 'path/to/doc1', 'This is a document.\n'),
            Document('doc2', 'path/to/doc2', 'This also is a document.\n')}
    doc_repo = DocumentFakeRepository(docs, 'eng', tmp_path)
    _preprocess_docs(doc_matcher, 'eng', True, doc_repo)
    assert doc_matcher.preprocessed_docs == docs


def test_preprocess_with_already_preprocessed_docs(tmp_path):
    doc_matcher = FakeDocumentMatcher()
    docs = {Document('doc1', 'path/to/doc1', 'This is a document.\n'),
            Document('doc2', 'path/to/doc2', 'This also is a document.\n')}
    doc_repo = DocumentFakeRepository(docs, 'eng', tmp_path)
    with patch.object(DocumentPickleRepository, 'list', return_value=docs):
        _preprocess_docs(doc_matcher, 'eng', True, doc_repo)
    assert doc_matcher.preprocessed_docs == set()


def test_preprocess_with_partly_preprocessed_docs(tmp_path):
    doc_matcher = FakeDocumentMatcher()
    docs = {Document('doc1', 'path/to/doc1', 'This is a document.\n'),
            Document('doc2', 'path/to/doc2', 'This also is a document.\n')}
    doc3 = Document('doc3', 'path/to/doc3', 'This is new.')
    doc_repo = DocumentFakeRepository({*docs, doc3}, 'eng', tmp_path)
    with patch.object(DocumentPickleRepository, 'list', return_value=docs):
        _preprocess_docs(doc_matcher, 'eng', True, doc_repo)
    assert doc_matcher.preprocessed_docs == {doc3}


@patch.object(DocumentPickleRepository, 'list', return_value={Document('doc1', 'path/to/doc1', 'This is a document.\n'),
                                                              Document('doc2', 'path/to/doc2', 'Another document.\n')})
@patch.object(DocumentPickleRepository, 'save')
def test_preprocess_serializes_new_document(save_mock, list_mock, tmp_path):
    doc_matcher = FakeDocumentMatcher()
    docs = {Document('doc1', 'path/to/doc1', 'This is a document.\n'),
            Document('doc2', 'path/to/doc2', 'Another document.\n')}
    doc3 = Document('doc3', 'path/to/doc3', 'This is new.')
    doc_repo = DocumentFakeRepository({*docs, doc3}, 'eng', tmp_path)
    _preprocess_docs(doc_matcher, 'eng', True, doc_repo)
    save_mock.assert_called_with({*docs, doc3})


@patch.object(DocumentPickleRepository, 'list', return_value={Document('doc1', 'path/to/doc1', 'This is a document.\n'),
                                                              Document('doc2', 'path/to/doc2', 'Another document.\n'),
                                                              Document('doc3', 'path/to/doc3', 'This is old.')})
@patch.object(DocumentPickleRepository, 'save')
def test_preprocess_removes_formerly_preprocessed_document(save_mock, list_mock, tmp_path):
    doc_matcher = FakeDocumentMatcher()
    docs = {Document('doc1', 'path/to/doc1', 'This is a document.\n'),
            Document('doc2', 'path/to/doc2', 'Another document.\n')}
    doc_repo = DocumentFakeRepository(docs, 'eng', tmp_path)
    _preprocess_docs(doc_matcher, 'eng', True, doc_repo)
    save_mock.assert_called_with(docs)


def test_find_matches(config, tmp_path):
    docs = [Document('doc1', 'path/to/doc1', 'This is a document.\n'),
            Document('doc2', 'path/to/doc2', 'This also is a document.\n')]
    doc_repo = DocumentFakeRepository(set(docs), 'eng', tmp_path)
    doc_pair_matches = DocumentPairMatches(docs[0], docs[1])
    doc_pair_matches.add(Match(MatchType.VERBATIM, Fragment(0, 18, docs[0]), Fragment(0, 23, docs[1])))
    with patch.object(DocumentMatcher, 'find_matches') as alg_fm:
        alg_fm.return_value = [doc_pair_matches]
        matches = find_matches(doc_repo, config=config)
    assert matches == [doc_pair_matches]
    alg_fm.assert_called_with(doc_repo.list(), None)


def test_find_matches_catches_unsupported_language_error(config, tmp_path):
    docs = [Document('doc1', 'path/to/doc1', 'This is a document.\n'),
            Document('doc2', 'path/to/doc2', 'This also is a document.\n')]
    doc_repo = DocumentFakeRepository(set(docs), 'eng', tmp_path)
    with patch.object(DocumentMatcher, 'find_matches') as alg_fm:
        alg_fm.side_effect = UnsupportedLanguageError()
        with pytest.raises(UsageError):
            find_matches(doc_repo, config=config)


def test_find_matches_catches_unsupported_file_format_error(config, tmp_path):
    docs = [Document('doc1', 'path/to/doc1', 'This is a document.\n'),
            Document('doc2', 'path/to/doc2', 'This also is a document.\n')]
    doc_repo = DocumentFakeRepository(set(docs), 'eng', tmp_path)
    with patch.object(DocumentFakeRepository, 'list', side_effect=UnsupportedFileFormatError()):
        with pytest.raises(UsageError):
            find_matches(doc_repo, config=config)


def test_find_matches_fails_on_unexpected_error(config, tmp_path):
    docs = [Document('doc1', 'path/to/doc1', 'This is a document.\n'),
            Document('doc2', 'path/to/doc2', 'This also is a document.\n')]
    doc_repo = DocumentFakeRepository(set(docs), 'eng', tmp_path)
    with patch.object(DocumentMatcher, 'find_matches') as alg_fm:
        alg_fm.side_effect = Exception()
        with pytest.raises(Exception):
            find_matches(doc_repo, config=config)


def test_write_xml_reports(matches, tmp_path):
    doc_pair_repo = DocumentPairMatchesJsonRepository(tmp_path)
    write_json_reports(matches, doc_pair_repo)
    assert len(doc_pair_repo.list()) == 2
