from __future__ import annotations

import os
from collections import Counter
from functools import partial
from urllib.parse import urlparse

import stanza
from stanza import Pipeline
from tqdm.contrib.concurrent import thread_map
from urlextract import URLExtract

from plagdef.model import stopwords
from plagdef.model.models import Document, Sentence, Word

PRCS = 'tokenize,mwt,pos,lemma'
PIPE_LVL = 'WARN'
LOAD_LVL = 'INFO'
LANG_CODES = {'ger': 'de', 'eng': 'en'}


class Preprocessor:
    def __init__(self, min_sent_len: int, rem_stop_words: bool):
        self._min_sent_len = min_sent_len
        self._rem_stop_words = rem_stop_words

    def preprocess(self, lang: str, docs: set[Document], common_docs: list[Document] = None):
        nlp_model = _nlp_pipe(lang)
        stop_words = stopwords.ENGLISH if lang == 'eng' else stopwords.GERMAN
        common_word_lists = _common_word_lists(nlp_model, common_docs) if common_docs else []
        thread_map(partial(self._preprocess, nlp_model=nlp_model, common_word_lists=common_word_lists,
                           stop_words=stop_words), docs, max_workers=os.cpu_count(),
                   total=len(docs), desc='Preprocessing', unit='doc')

    def _preprocess(self, doc: Document, nlp_model: Pipeline, common_word_lists: list[list[str]], stop_words: set[str]):
        sents = nlp_model(doc.text).sentences
        for sent_idx, sent in enumerate(sents):
            filtered_words = _word_filter(sent.words)
            if self._rem_stop_words:
                sent_lemmas = [word.lemma for word in filtered_words if word.text.lower() not in stop_words]
            else:
                sent_lemmas = [word.lemma for word in filtered_words]
            if len(sent_lemmas):
                lemma_count = Counter(sent_lemmas)
                sentence = Sentence(sent.tokens[0].start_char, sent.tokens[-1].end_char, lemma_count, doc)
                sentence.words = [Word(word.parent.start_char, word.parent.end_char, sentence)
                                  for word in filtered_words]
                doc.add_sent(sentence)
                if _sent_contains_common_words(sentence.words, common_word_lists):
                    sentence.common = True
                else:
                    for lemma in lemma_count.keys():
                        doc.vocab[lemma] += 1
        self._join_small_sentences(doc)
        self._remove_small_sentences(doc)
        _extract_urls(doc)

    def _join_small_sentences(self, doc: Document):
        sents = doc.sents(include_common=True)
        idx, sent_count = 0, len(sents)
        while idx < sent_count - 1:
            sent1, sent2 = sents[idx], sents[idx + 1]
            if (not sent1.common and not sent2.common) and \
                (len(sent1.words) < self._min_sent_len or
                 (sent2 == sents[-1] and len(sent2.words) < self._min_sent_len)):
                for lemma in sent1.bow.keys():
                    if lemma in sent2.bow:
                        doc.vocab[lemma] -= 1
                joined_sent = Sentence(sent1.start_char, sent2.end_char, sent1.bow + sent2.bow, doc)
                joined_sent.words = sent1.words + sent2.words
                doc.remove_sent(sent1), doc.remove_sent(sent2)
                doc.add_sent(joined_sent)
                idx -= 1
                sent_count -= 1
            idx += 1

    def _remove_small_sentences(self, doc: Document):
        sents = list(doc.sents())
        idx, sent_count = 0, len(sents)
        while idx < sent_count:
            if len(sents[idx].words) < self._min_sent_len:
                for lemma in sents[idx].bow.keys():
                    doc.vocab[lemma] -= 1
                doc.remove_sent(sents[idx]), sents.remove(sents[idx])
                idx -= 1
                sent_count -= 1
            idx += 1
        doc.vocab += Counter()  # Remove zero counts


def _extract_urls(doc: Document, extractor=URLExtract()):
    extractor.update_when_older(7)
    urls = extractor.find_urls(doc.text, only_unique=True, check_dns=True)
    doc.urls.update({urlparse(url, "https").geturl().rstrip("/").replace("///", "//") for url in urls})


def _nlp_pipe(lang: str) -> Pipeline:
    if lang in LANG_CODES:
        try:
            return stanza.Pipeline(LANG_CODES[lang], processors=PRCS, logging_level=PIPE_LVL)
        except:  # Unpickling error raises Exception, cannot narrow
            stanza.download(LANG_CODES[lang], processors=PRCS, logging_level=LOAD_LVL)
            return stanza.Pipeline(LANG_CODES[lang], processors=PRCS, logging_level=PIPE_LVL)
    else:
        raise UnsupportedLanguageError(f'The language "{lang}" is currently not supported.')


def _common_word_lists(pipe: Pipeline, common_docs: list[Document]) -> list[list[str]]:
    common_word_lists = []
    for doc in common_docs:
        for line in doc.text.splitlines():
            parsed_line = pipe(line)
            line_words = []
            for sent in parsed_line.sentences:
                [line_words.append(word.text.lower()) for word in _word_filter(sent.words)]
            common_word_lists.append(line_words) if len(line_words) else None
    return common_word_lists


def _word_filter(stanza_words: list) -> list:
    return [word for word in stanza_words if not word.upos == 'PUNCT' and word.text.isalnum() and len(word.text) > 1]


def _sent_contains_common_words(sent_words: list[Word], common_word_lists: list[list[str]]) -> bool:
    sent_word_texts = [word.text.lower() for word in sent_words]
    for common_word_list in common_word_lists:
        if all(common_word in sent_word_texts for common_word in common_word_list):
            return True
    return False


class UnsupportedLanguageError(Exception):
    pass
