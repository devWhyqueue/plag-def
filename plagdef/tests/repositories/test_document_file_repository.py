from fpdf import FPDF

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


# TODO: Implement
def test_list_ignores_unsupported_binary_files():
    pass


def test_pdf_reader_poor_extraction():
    text = 'Ein w(cid:246)rtlicher Match.'
    reader = PdfReader(None, lang='ger', use_ocr=True)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_mark_umlaut():
    text = 'Ein w¨ortlicher Match.'
    reader = PdfReader(None, lang='ger', use_ocr=True)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_ff():
    text = 'Eine fehlerhafte Veröﬀentlichung.'
    reader = PdfReader(None, lang='ger', use_ocr=True)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_very_long_word():
    text = 'TheseWordsarewrongfullymergedtogetherduetoextractionproblems.'
    reader = PdfReader(None, lang='eng', use_ocr=True)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_url():
    text = ' https://www.treatwell.de/partners/inspiration/blog/5-tipps-ihr-team-zu-motivieren'
    reader = PdfReader(None, lang='eng', use_ocr=True)
    assert not reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_no_text():
    text = '   '
    reader = PdfReader(None, lang='eng', use_ocr=True)
    assert reader._poor_extraction(text)


def test_pdf_reader_poor_extraction_with_correct_text():
    text = 'This is flawless.'
    reader = PdfReader(None, lang='eng', use_ocr=True)
    assert not reader._poor_extraction(text)


def test_pdf_reader_merges_hyphenated_words_at_line_end(tmp_path):
    doc1 = FPDF()
    doc1.add_page()
    doc1.set_font('helvetica', size=12)
    doc1.multi_cell(0, 5, 'This is a PDF file con-\n'
                          'taining one sentence. However there are mul- \n'
                          'tiple line breaks which split words.')
    doc1.output(f'{tmp_path}/doc1.pdf')
    reader = PdfReader(tmp_path / 'doc1.pdf', lang='eng', use_ocr=True, )
    text = reader._extract()
    assert text == 'This is a PDF file containing one sentence. However there are multiple line breaks' \
                   ' which split words.'
