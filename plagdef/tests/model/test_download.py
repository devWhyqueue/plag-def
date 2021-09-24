from unittest.mock import patch

import requests
from bs4 import BeautifulSoup
from requests.exceptions import SSLError

from plagdef.model.download import _download_page, download_external_sources, download_all_external_sources
from plagdef.model.models import Document
from plagdef.tests.fakes import FakeResponse


@patch("requests.get", return_value=FakeResponse({'content-type': 'text/html'}, b'Google Suche', 'Google Suche'))
@patch.object(BeautifulSoup, 'get_text', return_value='Google Suche')
def test_download_page(req_mock, bs_mock, tmp_path):
    file = _download_page("https://google.com", tmp_path)
    assert bs_mock.is_called()
    assert file.path.name == "google.com.txt"
    assert file.content == "Google Suche"
    assert not file.binary


@patch("requests.get", return_value=FakeResponse({'content-type': 'text/html'}, b'', ''))
@patch.object(BeautifulSoup, 'get_text', return_value='')
def test_download_page_returns_none_if_content_length_zero(req_mock, bs_mock, tmp_path):
    file = _download_page("https://google.com", tmp_path)
    assert not file


@patch("requests.get", return_value=FakeResponse({'content-type': 'text/html'}, b'Content', 'Content'))
@patch.object(BeautifulSoup, 'get_text', return_value='Content')
def test_download_page_with_url_with_path(req_mock, bs_mock, tmp_path):
    file = _download_page("https://reisevergnuegen.com/deutschland-duesseldorf-tipps", tmp_path)
    assert file.path.name == "deutschland-duesseldorf-tipps_from_reisevergnuegen.com.txt"


@patch("requests.get", return_value=FakeResponse({'content-type': 'text/html'}, b'Content', 'Content'))
@patch.object(BeautifulSoup, 'get_text', return_value='Content')
def test_download_page_with_url_containing_insecure_chars(req_mock, bs_mock, tmp_path):
    file = _download_page("https://google.de/*!<<<>|a", tmp_path)
    assert file.path.name == "a_from_google.de.txt"


@patch("requests.get", return_value=FakeResponse({'content-type': 'application/pdf'}, b'Content', 'Content'))
def test_download_page_with_different_content_type(req_mock, tmp_path):
    file = _download_page("https://google.de", tmp_path)
    assert file.path.name == "google.de.pdf"
    assert file.binary


@patch("requests.get", return_value=FakeResponse({}, b'Content', 'Content'))
def test_download_page_with_no_content_type(req_mock, tmp_path):
    file = _download_page("https://google.de", tmp_path)
    assert file.path.name == "google.de.txt"
    assert not file.binary


@patch("requests.get", side_effect=requests.ConnectionError())
def test_download_page_catches_timeout_error(req_mock, tmp_path):
    file = _download_page("https://google.de", tmp_path)
    assert not file


@patch("requests.get", side_effect=SSLError())
def test_download_page_catches_ssl_error(req_mock, tmp_path):
    file = _download_page("https://google.de", tmp_path)
    assert not file


@patch("requests.get", side_effect=SSLError())
def test_download_external_sources_filters_none(req_mock, tmp_path):
    doc = Document("doc", "path/to/doc", "Google.com is the most popular search engine.")
    doc.urls = {"https://google.com"}
    files = download_external_sources(doc, tmp_path)
    assert files == set()


@patch("requests.get", return_value=FakeResponse({'content-type': 'text/html'}, b'Content', 'Content'))
@patch.object(BeautifulSoup, 'get_text', return_value='Content')
def test_download_external_sources_filters_by_same_contents(req_mock, bs_mock, tmp_path):
    doc1 = Document("doc1", "path/to/doc1", "https://website.com, https://samewebsite.com")
    doc1.urls = {"https://website.com", "https://samewebsite.com"}
    files = download_external_sources(doc1, tmp_path)
    assert len(files) == 1


@patch("requests.get", side_effect=[FakeResponse({'content-type': 'text/html'}, b'Google...', 'Google...'),
                                    FakeResponse({'content-type': 'text/html'}, b'Bing...', 'Bing...')])
@patch.object(BeautifulSoup, 'get_text', side_effect=['Google...', 'Bing...'])
def test_download_all_external_sources(req_mock, bs_mock, tmp_path):
    doc1 = Document("doc1", "path/to/doc1", "Google.com is the most popular search engine.")
    doc1.urls = {"https://google.com"}
    doc2 = Document("doc2", "path/to/doc2", "Bing.com is the a popular search engine.")
    doc2.urls = {"https://bing.com"}
    files = download_all_external_sources({doc1, doc2}, tmp_path)
    assert len(files) == 2
    assert "google.com.txt", "bing.com.txt" in [file.path.name for file in files]


@patch("requests.get", return_value=FakeResponse({'content-type': 'text/html'}, b'Content', 'Content'))
@patch.object(BeautifulSoup, 'get_text', return_value='Content')
def test_download_all_external_sources_filters_by_same_contents(req_mock, bs_mock, tmp_path):
    doc1 = Document("doc1", "path/to/doc1", "https://website.com")
    doc1.urls = {"https://website.com"}
    doc2 = Document("doc2", "path/to/doc2", "https://samewebsite.com")
    doc2.urls = {"https://samewebsite.com"}
    files = download_all_external_sources({doc1, doc2}, tmp_path)
    assert len(files) == 1
