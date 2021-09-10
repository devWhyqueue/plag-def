from os import urandom
from pathlib import Path
from unicodedata import normalize

import pytest
from fpdf import FPDF

from plagdef.model.models import File
from plagdef.repositories import FileRepository


def test_init_with_nonexistent_doc_dir_fails():
    with pytest.raises(NotADirectoryError):
        FileRepository(Path('some/wrong/path'))


def test_init_with_file_fails(tmp_path):
    file = tmp_path / 'doc1.txt'
    with file.open('w', encoding='utf-8') as f:
        f.write('Some content.\n')
    with pytest.raises(NotADirectoryError):
        FileRepository(file)


def test_list(tmp_path):
    doc1 = FPDF()
    doc1.add_page()
    doc1.set_font('helvetica', size=12)
    doc1.cell(w=0, txt='This is a PDF file containing one sentence.')
    doc1.output(f'{tmp_path}/doc1.pdf')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('This is a text file.')
    Path(f'{tmp_path}/a/dir/that/should/not/be/included').mkdir(parents=True)
    with Path(f'{tmp_path}/a/dir/that/should/not/be/included/doc3.txt').open('w', encoding='utf-8') as f:
        f.write('The third file.\n')
    repo = FileRepository(tmp_path)
    files = repo.list()
    assert len(files) == 2
    assert files == {File(Path(f'{tmp_path}/doc1.pdf'), Path(f'{tmp_path}/doc1.pdf').read_bytes(), True),
                     File(Path(tmp_path / 'doc2.txt'), "This is a text file.", False)}


def test_list_normalizes_texts(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='utf-8') as f:
        # The 'ä' consists of a small letter a and a combining diaeresis
        f.write('Nicht nur ähnlich, sondern gleich.')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('Nicht nur ähnlich, sondern gleich.')
    repo = FileRepository(tmp_path)
    file1, file2 = repo.list()
    assert file1.content == file2.content


def test_list_removes_bom(tmp_path):
    with (tmp_path / 'doc1.txt').open('wb') as f:
        f.write(b'\xef\xbb\xbfFiles should be identical although one starts with BOM.')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('Files should be identical although one starts with BOM.')
    repo = FileRepository(tmp_path)
    file1, file2 = repo.list()
    assert file1.content == file2.content


def test_list_with_file_containing_special_characters(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='utf-8') as f:
        f.write('These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    repo = FileRepository(tmp_path)
    files = repo.list()
    assert len(files) == 2
    assert 'These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n' \
           in [file.content for file in files]


@pytest.mark.skipif('sys.platform != "win32"')
def test_list_with_dir_containing_ansi_file(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='ANSI') as f:
        f.write('These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    repo = FileRepository(tmp_path)
    files = repo.list()
    assert len(files) == 2
    assert 'These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n' \
           in [file.content for file in files]


def test_list_with_doc_dir_containing_iso_8559_1_file(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='ISO-8859-1') as f:
        f.write('These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n')
    with (tmp_path / 'doc2.txt').open('w', encoding='utf-8') as f:
        f.write('Hello world, Καλημέρα κόσμε, コンニチハ')
    repo = FileRepository(tmp_path)
    files = repo.list()
    assert len(files) == 2
    assert 'These are typical German umlauts: ä, ö, ü, ß, é and â are rather French.\n' in [file.content for file in
                                                                                            files]
    assert normalize('NFC', 'Hello world, Καλημέρα κόσμε, コンニチハ') in [file.content for file in files]


def test_list_recursive(tmp_path):
    with (tmp_path / 'doc1.txt').open('w', encoding='utf-8') as f:
        f.write('This is a document.')
    Path(f'{tmp_path}/a/subdir').mkdir(parents=True)
    with Path((f'{tmp_path}/a/subdir/doc2.txt')).open('w', encoding='utf-8') as f:
        f.write('This also is a document.\n')
    Path(f'{tmp_path}/another/sub/even/deeper').mkdir(parents=True)
    with Path((f'{tmp_path}/another/sub/even/deeper/doc3.txt')).open('w', encoding='utf-8') as f:
        f.write('The third document.\n')
    repo = FileRepository(tmp_path, recursive=True)
    files = repo.list()
    assert len(files) == 3


def test_save(tmp_path):
    file = File(Path(tmp_path / "doc.txt"), "Hello World!", False)
    file_repo = FileRepository(tmp_path)
    file_repo.save(file)
    assert Path(tmp_path / "doc.txt").read_text() == "Hello World!"


def test_save_binary(tmp_path):
    content = urandom(128)
    file = File(Path(tmp_path / "doc.pdef"), content, True)
    file_repo = FileRepository(tmp_path)
    file_repo.save(file)
    assert Path(tmp_path / "doc.pdef").read_bytes() == content


def test_save_if_file_exists(tmp_path):
    file = File(Path(tmp_path / "doc.txt"), "Hello World!", False)
    file_repo = FileRepository(tmp_path)
    file_repo.save(file)
    with pytest.raises(FileExistsError):
        file_repo.save(File(Path(tmp_path / "doc.txt"), "Other content.", False))
    assert Path(tmp_path / "doc.txt").read_text() == "Hello World!"


def test_save_all(tmp_path):
    files = {File(Path(tmp_path / "doc1.txt"), "Hello World!", False),
             File(Path(tmp_path / "doc2.txt"), "Heyho World!", False)}
    file_repo = FileRepository(tmp_path)
    file_repo.save_all(files)
    assert len(list(Path(tmp_path).iterdir())) == 2
