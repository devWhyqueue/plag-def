from pathlib import Path
from unicodedata import normalize

import pytest
from fpdf import FPDF

from plagdef.repositories import DocumentFileRepository, PdfReader


def test_init_with_nonexistent_doc_dir_fails():
    with pytest.raises(NotADirectoryError):
        DocumentFileRepository(Path('some/wrong/path'), lang='eng', use_ocr=True)


def test_init_with_file_fails(tmp_path):
    file = tmp_path / 'doc1.txt'
    with file.open('w', encoding='utf-8') as f:
        f.write('Some content.\n')
    with pytest.raises(NotADirectoryError):
        DocumentFileRepository(file, lang='eng', use_ocr=True)


def test_list_documents(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='utf-8') as f:
        f.write('This is a document.\n')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    Path(f'{tmp_path}/a/dir/that/should/not/be/included').mkdir(parents=True)
    with Path((f'{tmp_path}/a/dir/that/should/not/be/included/doc3.txt')).open('w', encoding='utf-8') as f:
        f.write('The third document.\n')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    docs = repo.list()
    assert len(docs) == 2


def test_list_documents_ignores_pdef_files(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='utf-8') as f:
        f.write('This is a document.\n')
    with (tmp_path / 'prep.pdef').open('w', encoding='utf-8') as f:
        f.write('This is a preprocessing file.')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    docs = repo.list()
    assert len(docs) == 1


def test_list_documents_normalizes_texts(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='utf-8') as f:
        # The 'ä' consists of a small letter a and a combining diaeresis
        f.write('Nicht nur ähnlich, sondern gleich.')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('Nicht nur ähnlich, sondern gleich.')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    doc1, doc2 = repo.list()
    assert doc1.text == doc2.text


def test_list_documents_removes_bom(tmp_path):
    with (tmp_path / 'doc1.txt').open('wb') as f:
        f.write(b'\xef\xbb\xbfDocuments should be identical although one starts with BOM.')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('Documents should be identical although one starts with BOM.')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    doc1, doc2 = repo.list()
    assert doc1.text == doc2.text


def test_list_with_file_containing_special_characters(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='utf-8') as f:
        f.write('These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    docs = repo.list()
    assert len(docs) == 2
    assert 'These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n' in [doc.text for doc in docs]


@pytest.mark.skipif('sys.platform != "win32"')
def test_list_with_doc_dir_containing_ansi_file_creates_document(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='ANSI') as f:
        f.write('These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    docs = repo.list()
    assert len(docs) == 2
    assert 'These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n' in [doc.text for doc in docs]


def test_list_with_doc_dir_containing_iso_8559_1_file_creates_documents(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='ISO-8859-1') as f:
        f.write('These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('Hello world, Καλημέρα κόσμε, コンニチハ')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True)
    docs = repo.list()
    assert len(docs) == 2
    assert 'These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n' in [doc.text for doc in docs]
    assert normalize('NFC', 'Hello world, Καλημέρα κόσμε, コンニチハ') in [doc.text for doc in docs]


def test_list_recursive_creates_documents(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='utf-8') as f:
        f.write('This is a document.\n')
    Path(f'{tmp_path}/a/subdir').mkdir(parents=True)
    with Path((f'{tmp_path}/a/subdir/doc2.txt')).open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    Path(f'{tmp_path}/another/sub/even/deeper').mkdir(parents=True)
    with Path((f'{tmp_path}/another/sub/even/deeper/doc3.txt')).open('w', encoding='utf-8') as f:
        f.write('The third document.\n')
    repo = DocumentFileRepository(tmp_path, lang='eng', use_ocr=True, recursive=True)
    docs = repo.list()
    assert len(docs) == 3


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
