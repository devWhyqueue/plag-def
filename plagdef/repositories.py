from __future__ import annotations

import bz2
import logging
import os
import re
from collections import Counter
from copy import deepcopy
from hashlib import blake2b
from io import BytesIO
from json import JSONDecodeError
from multiprocessing import Lock
from pathlib import Path
from pickle import dump, load, UnpicklingError
from unicodedata import normalize

import jsonpickle
import magic
import pdfplumber
from magic import MagicException
from ocrmypdf import ocr
from sortedcontainers import SortedSet
from tqdm.contrib.concurrent import process_map

from plagdef.config import settings
from plagdef.model import models

log = logging.getLogger(__name__)
jsonpickle.set_encoder_options('json', indent=4)
lock = Lock()


class FileRepository:
    def __init__(self, dir_path: Path, recursive=False):
        self.dir_path = dir_path
        self._recursive = recursive
        if not dir_path.is_dir():
            raise NotADirectoryError(f'The given path {dir_path} does not point to an existing directory!')

    def list(self) -> set[models.File]:
        files = set()
        f_gen = self.dir_path.rglob('*') if self._recursive else self.dir_path.iterdir()
        for f in f_gen:
            if f.is_file():
                binary = not magic.Magic(mime=True).from_buffer(open(f, 'rb').read(2048)).startswith("text")
                try:
                    content = f.read_bytes() if binary else self._read_text(f)
                    files.add(models.File(f.name, content, binary))
                except UnsupportedFileFormatError as e:
                    log.error(e)
                    log.debug('Following error occurred:', exc_info=True)
        return files

    def _read_text(self, file_path: Path):
        try:
            detect_enc = magic.Magic(mime_encoding=True)
            enc = detect_enc.from_buffer(open(file_path, 'rb').read(2048))
            enc_str = enc if enc != 'utf-8' else 'utf-8-sig'
            text = file_path.read_text(encoding=enc_str)
            return normalize('NFC', text)
        except (UnicodeDecodeError, LookupError, MagicException):
            raise UnsupportedFileFormatError(
                f"The file '{file_path.name}' has an unsupported encoding and cannot be read.")

    def save(self, file: models.File):
        file_path = Path(f"{self.dir_path}/{file.name}")
        if file_path.exists():
            raise FileExistsError(f'The file "{file_path}" already exists!')
        if file.binary:
            with Path(file_path).open('wb') as f:
                f.write(file.content)
        else:
            with Path(file_path).open('w', encoding="utf-8") as f:
                f.write(file.content)

    def save_all(self, files: set[models.File]):
        for file in files:
            try:
                self.save(file)
            except FileExistsError:
                log.warning(f'Ignoring already existing external source "{file.name}".')


class DocumentFileRepository:
    def __init__(self, dir_path: Path, recursive=False, lang=None, use_ocr=None):
        self._file_repo = FileRepository(dir_path, recursive)
        self.lang = lang if lang else settings['lang']
        self._use_ocr = use_ocr if use_ocr else settings['ocr']

    @property
    def dir_path(self):
        return self._file_repo.dir_path

    def list(self) -> set[models.Document]:
        files = list(filter(lambda f: not f.name.lower().endswith(".pdef"), self._file_repo.list()))
        docs = process_map(self._create_doc, files, desc=f"Reading documents in '{self.dir_path}'",
                           unit='doc', total=len(files), max_workers=os.cpu_count())
        return set(filter(None, docs))

    def _create_doc(self, file: models.File) -> models.Document:
        name = file.name.rsplit(".", 1)[0]
        doc_path = Path(f"{self.dir_path}/{file.name}").resolve()
        if file.name.lower().endswith('.pdf'):
            reader = PdfReader(doc_path, self.lang, self._use_ocr)
            text = reader.extract_text()
            doc = models.Document(name, str(doc_path), text)
            doc.urls = reader.extract_urls()
        elif not file.binary:
            doc = models.Document(name, str(doc_path), file.content)
        else:
            log.warning(f'Ignoring unsupported binary file "{file.name}".')
            doc = None
        return doc


class DocumentPairRepository:
    def __init__(self, doc1: models.Document, doc2: models.Document):
        self._docs = {doc1, doc2}

    def list(self) -> set[models.Document]:
        return self._docs


class DocumentPairMatchesJsonRepository:
    def __init__(self, out_path: Path):
        if not out_path.is_dir():
            raise NotADirectoryError(f"The given path '{out_path}' does not point to an existing directory!")
        self._out_path = out_path

    def save(self, doc_pair_matches):
        clone = deepcopy(doc_pair_matches)
        clone.doc1.vocab, clone.doc2.vocab = Counter(), Counter()
        clone.doc1._sents, clone.doc2._sents = SortedSet(), SortedSet()
        file_name = Path(f'{clone.doc1.name}-{clone.doc2.name}.json')
        file_path = self._out_path / file_name
        with file_path.open('w', encoding='utf-8') as f:
            text = jsonpickle.encode(clone)
            f.write(text)

    def list(self) -> set[models.DocumentPairMatches]:
        doc_pair_matches_list = set()
        for file in self._out_path.iterdir():
            if file.is_file() and file.suffix == '.json':
                try:
                    text = file.read_text(encoding='utf-8')
                    doc_pair_matches = jsonpickle.decode(text)
                    doc_pair_matches_list.add(doc_pair_matches)
                except (UnicodeDecodeError, JSONDecodeError):
                    log.error(f"The file '{file.name}' could not be read.")
                    log.debug('Following error occurred:', exc_info=True)
        return doc_pair_matches_list


class DocumentPickleRepository:
    def __init__(self, dir_path: Path, common_dir_path: Path = None):
        if not dir_path.is_dir():
            raise NotADirectoryError(f"The given path '{dir_path}' does not point to an existing directory!")
        path_hash = blake2b(str(common_dir_path).encode(), digest_size=16).hexdigest()
        self.file_path = dir_path / f'.{path_hash}.pdef'

    def save(self, docs: set[models.Document]):
        with bz2.open(self.file_path, 'wb') as file:
            dump(docs, file)

    def list(self) -> set[models.Document]:
        if self.file_path.exists():
            log.info('Found preprocessing file. Deserializing...')
            try:
                with bz2.open(self.file_path, 'rb') as file:
                    return load(file)
            except (UnpicklingError, EOFError):
                log.warning(f"Could not deserialize preprocessing file, '{self.file_path.name}' seems to be corrupted.")
                log.debug('Following error occurred:', exc_info=True)
        return set()


class PdfReader:
    ERROR_HEURISTIC = '¨[aou]|ﬀ|\(cid:\d+\)|[a-zA-Z]{50}'

    def __init__(self, file, lang, use_ocr):
        self._file = file
        self._lang = lang if lang == 'eng' else 'deu'
        self._use_ocr = use_ocr

    def extract_urls(self):
        with pdfplumber.open(self._file) as pdf:
            return {uri_obj['uri'] for uri_obj in pdf.hyperlinks}

    def extract_text(self):
        text = self._extract()
        if self._use_ocr and self._poor_extraction(text):
            log.warning(f"Poor text extraction in '{self._file.name}' detected! Using OCR...")
            with BytesIO() as ocr_file:
                with lock:
                    ocr(self._file, ocr_file, language=self._lang, force_ocr=True, progress_bar=False,
                        max_image_mpixels=512)
                text = self._extract(ocr_file)
        return text

    def _extract(self, file=None) -> str:
        if file is None:
            file = self._file
        with pdfplumber.open(file) as pdf:
            text = ' '.join(filter(None, (page.extract_text() for page in pdf.pages)))
            normalized_text = normalize('NFC', text)
            return re.sub('-\s?\n', '', normalized_text)  # Merge hyphenated words

    def _poor_extraction(self, text: str) -> bool:
        return not len(text.strip()) or bool(re.search(PdfReader.ERROR_HEURISTIC, text))


class UnsupportedFileFormatError(Exception):
    pass
