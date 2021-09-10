from __future__ import annotations

import logging
import os
from functools import partial
from mimetypes import guess_extension
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.exceptions import SSLError
from tqdm.contrib.concurrent import thread_map
from urllib3.exceptions import MaxRetryError
from werkzeug.utils import secure_filename

from plagdef.model.models import Document, File

log = logging.getLogger(__name__)


def download_all_external_sources(docs: set[Document], target_dir: Path) -> set[File]:
    urls = {url for doc in docs for url in doc.urls}
    return set(filter(None, thread_map(partial(_download_page, target_dir=target_dir), urls, max_workers=os.cpu_count(),
                                       total=len(urls), desc='Downloading', unit='external sources')))


def download_external_sources(doc: Document, target_dir: Path) -> set[File]:
    return set(filter(None, thread_map(partial(_download_page, target_dir=target_dir), doc.urls,
                                       max_workers=os.cpu_count(), total=len(doc.urls), desc='Downloading',
                                       unit='external sources')))


def _download_page(url: str, target_dir: Path) -> File:
    parsed_url = urlparse(url)
    filename = parsed_url.netloc if not parsed_url.path else \
        f'{parsed_url.path[parsed_url.path.rindex("/") + 1:]}_from_{parsed_url.netloc}'
    filename = secure_filename(filename)
    try:
        resp = requests.get(url)
        mime_type = resp.headers['content-type']
        binary = not mime_type.startswith('text')
        content = resp.content if binary else resp.text
        ext = guess_extension(mime_type)
        if mime_type.startswith("text/html"):
            content = BeautifulSoup(resp.text, features="lxml").get_text(" ", True)
            ext = ".txt"
        file = File(Path(f"{target_dir}/{filename}{ext}"), content, binary)
        return file
    except (MaxRetryError, SSLError):
        log.warning(f'Could not download from "{url}".')
