from __future__ import annotations

import bz2
import logging
import os
import re
from collections import Counter
from copy import deepcopy
from functools import partial
from hashlib import blake2b
from json import JSONDecodeError
from multiprocessing import Lock
from pathlib import Path
from pickle import dump, load, UnpicklingError
from unicodedata import normalize

import jsonpickle
import magic
import numpy
import pdfplumber
from easyocr import easyocr
from magic import MagicException
from pdf2image import convert_from_path
from sortedcontainers import SortedSet
from tqdm.contrib.concurrent import thread_map

from plagdef.config import settings
from plagdef.model import models

log = logging.getLogger(__name__)
jsonpickle.set_encoder_options('json', indent=4)


class FileRepository:
    def __init__(self, base_path: Path, recursive=False):
        self.base_path = base_path
        self._recursive = recursive
        if not base_path.is_dir():
            raise NotADirectoryError(f'The given path {base_path} does not point to an existing directory!')

    def list(self) -> set[models.File]:
        files = set()
        f_gen = self.base_path.rglob('*') if self._recursive else self.base_path.iterdir()
        for f in f_gen:
            if f.is_file():
                binary = not magic.Magic(mime=True).from_buffer(open(f, 'rb').read(2048)).startswith("text")
                try:
                    content = f.read_bytes() if binary else self._read_text(f)
                    files.add(models.File(f, content, binary))
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
        if file.path.exists():
            raise FileExistsError(f'The file "{file.path.name}" already exists!')
        if file.binary:
            with file.path.open('wb') as f:
                f.write(file.content)
        else:
            with file.path.open('w', encoding="utf-8") as f:
                f.write(file.content)

    def save_all(self, files: set[models.File]):
        for file in files:
            try:
                self.save(file)
            except FileExistsError as e:
                log.debug(e)


class DocumentFileRepository:
    def __init__(self, dir_path: Path, recursive=False, lang=None, use_ocr=None):
        self._file_repo = FileRepository(dir_path, recursive)
        self.lang = lang if lang else settings['lang']
        enable_ocr = use_ocr if use_ocr else settings['ocr']
        self._ocr = easyocr.Reader(['de', 'en']) if enable_ocr else None

    @property
    def base_path(self):
        return self._file_repo.base_path

    def list(self) -> set[models.Document]:
        files = list(filter(lambda f: not f.path.suffix.lower() == '.pdef', self._file_repo.list()))
        read_file = partial(self._create_doc, lock=Lock())
        docs = thread_map(read_file, files, desc=f"Reading documents in '{self.base_path}'",
                          unit='doc', total=len(files), max_workers=os.cpu_count())
        return set(filter(None, docs))

    def _create_doc(self, file, lock):
        if file.path.suffix.lower() == '.pdf':
            reader = PdfReader(file.path, self._ocr, lock)
            text = reader.extract_text()
            doc = models.Document(file.path.stem, str(file.path), text)
            urls = reader.extract_urls()
            doc.urls.update(urls) if urls else None
        elif not file.binary:
            doc = models.Document(file.path.stem, str(file.path), file.content)
        else:
            log.warning(f'Ignoring unsupported file "{file.path.name}", as it cannot be read.')
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

    def __init__(self, file, ocr, lock):
        self._file = file
        self._ocr = ocr
        self._lock = lock

    def extract_urls(self):
        # Temporary fix for: https://github.com/jsvine/pdfplumber/issues/463
        try:
            with pdfplumber.open(self._file) as pdf:
                return {uri_obj['uri'].rstrip('/') for uri_obj in pdf.hyperlinks}
        except UnicodeDecodeError:
            log.warning(f'Could not extract hyperlinks from PDF "{self._file.name}".')

    def extract_text(self):
        text = self._extract()
        if self._ocr and self._poor_extraction(text):
            log.warning(f"Poor text extraction in '{self._file.name}' detected! Using OCR...")
            pages = convert_from_path(self._file, fmt='jpeg')
            text = ''
            for page in pages:
                img = numpy.array(page)
                with self._lock:
                    page_text = ' '.join(self._ocr.readtext(img, detail=0, paragraph=True))
                text += page_text
        return self._normalize_text(text)

    def _extract(self, file=None) -> str:
        if file is None:
            file = self._file
        with pdfplumber.open(file) as pdf:
            text = ' '.join(filter(None, (page.extract_text() for page in pdf.pages)))
            return self._normalize_text(text)

    def _normalize_text(self, text: str) -> str:
        normalized_text = normalize('NFC', text)
        return re.sub('-\s?\n', '', normalized_text)  # Merge hyphenated words

    def _poor_extraction(self, text: str) -> bool:
        return not len(text.strip()) or bool(re.search(PdfReader.ERROR_HEURISTIC, text))


class UnsupportedFileFormatError(Exception):
    pass
