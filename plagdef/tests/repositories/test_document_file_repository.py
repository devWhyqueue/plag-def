from pathlib import Path
from unittest.mock import patch

from fpdf import FPDF

from plagdef.model.models import File, Document
from plagdef.repositories import DocumentFileRepository, PdfReader


def test_list_documents_ignores_pdef_files(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='utf-8') as f:
        f.write('This is a document.\n')
    with (tmp_path / 'prep.pdef').open('w', encoding='utf-8') as f:
        f.write('This is a preprocessing file.')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    docs = repo.list()
    assert len(docs) == 1


def test_list_with_doc_dir_containing_pdf(tmp_path):
    doc1 = FPDF()
    doc1.add_page()
    doc1.set_font('helvetica', size=12)
    doc1.cell(w=0, txt='This is a PDF file containing one sentence.')
    doc1.output(f'{tmp_path}/doc1.pdf')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    docs = repo.list()
    assert 'This is a PDF file containing one sentence.' in [doc.text for doc in docs]
    assert 'This also is a document.\n' in [doc.text for doc in docs]


def test_list_with_doc_dir_containing_pdf_with_uppercase_file_suffix(tmp_path):
    doc1 = FPDF()
    doc1.add_page()
    doc1.set_font('helvetica', size=12)
    doc1.cell(w=0, txt='This is a PDF file containing one sentence.')
    doc1.output(f'{tmp_path}/doc1.PDF')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    docs = repo.list()
    assert 'This is a PDF file containing one sentence.' in [doc.text for doc in docs]
    assert 'This also is a document.\n' in [doc.text for doc in docs]


def test_list_with_doc_dir_containing_pdf_with_no_text(tmp_path):
    doc1 = FPDF()
    doc1.add_page()
    doc1.output(f'{tmp_path}/doc1.pdf')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    docs = repo.list()
    assert len(docs) == 2
    assert 'This also is a document.\n' in [doc.text for doc in docs]


def test_create_doc_with_file_in_subdir(tmp_path):
    doc_repo = DocumentFileRepository(tmp_path)
    # This file is located in tmp_path/sub/dir/doc.txt
    file_path = Path(f"{tmp_path}/sub/dir/doc.txt")
    file = File(file_path, "Hello World!", False)
    doc = doc_repo._create_doc(file)
    assert doc == Document("doc", str(file_path), "Hello World!")
    assert doc.path == str(file_path)


@patch.object(PdfReader, 'extract_text', return_value="Hello World!")
@patch.object(PdfReader, 'extract_urls', return_value=None)
def test_create_doc_if_extract_urls_returns_none(pdf_mock, tmp_path):
    doc_repo = DocumentFileRepository(tmp_path)
    file_path = Path(f"{tmp_path}/sub/dir/doc.pdf")
    file = File(file_path, b"Hello World!", True)
    doc = doc_repo._create_doc(file)
    assert doc.urls == set()


def test_pdf_reader_poor_extraction():
    text = 'Ein w(cid:246)rtlicher Match.'
    reader = PdfReader(None, None, None)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_mark_umlaut():
    text = 'Ein w¨ortlicher Match.'
    reader = PdfReader(None, None, None)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_ff():
    text = 'Eine fehlerhafte Veröﬀentlichung.'
    reader = PdfReader(None, None, None)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_very_long_word():
    text = 'TheseWordsarewrongfullymergedtogetherduetoextractionproblems.'
    reader = PdfReader(None, None, None)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_url():
    text = ' https://www.treatwell.de/partners/inspiration/blog/5-tipps-ihr-team-zu-motivieren'
    reader = PdfReader(None, None, None)
    assert not reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_no_text():
    text = '   '
    reader = PdfReader(None, None, None)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_with_correct_text():
    text = 'This is flawless.'
    reader = PdfReader(None, None, None)
    assert not reader._poor_extraction(text)


def test_pdf_reader_merges_hyphenated_words_at_line_end(tmp_path):
    doc1 = FPDF()
    doc1.add_page()
    doc1.set_font('helvetica', size=12)
    doc1.multi_cell(0, 5, 'This is a PDF file con-\n'
                          'taining one sentence. However there are mul- \n'
                          'tiple line breaks which split words.')
    doc1.output(f'{tmp_path}/doc1.pdf')
    reader = PdfReader(tmp_path / 'doc1.pdf', None, None)
    text = reader._extract()
    assert text == 'This is a PDF file containing one sentence. However there are multiple line breaks' \
                   ' which split words.'


@patch("pdfplumber.open", side_effect=UnicodeDecodeError("", bytes(), -1, -1, ""))
def test_pdf_reader_extract_urls_returns_none_on_unicode_decode_error(pdf_mock, tmp_path):
    reader = PdfReader(tmp_path, lang='eng', use_ocr=True)
    urls = reader.extract_urls()
    assert not urls
