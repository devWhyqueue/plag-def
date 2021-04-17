from __future__ import annotations

import logging
from collections import Counter

import stanza
from stanza import Pipeline
from tqdm import trange

from plagdef.model import stopwords
from plagdef.model.models import Document, Sentence, Word

log = logging.getLogger(__name__)

PRCS = 'tokenize,mwt,pos,lemma'
PIPE_LVL = 'WARN'
LOAD_LVL = 'INFO'
LANG_CODES = {'ger': 'de', 'eng': 'en'}


class Preprocessor:
    def __init__(self, min_sent_len: int, rem_stop_words: bool):
        self._min_sent_len = min_sent_len
        self._rem_stop_words = rem_stop_words

    def preprocess(self, lang: str, docs: list[Document], common_docs: list[Document] = None):
        log.info('Preprocessing documents...')
        nlp_model = _nlp_pipe(lang)
        stop_words = stopwords.ENGLISH if lang == 'eng' else stopwords.GERMAN
        # Preprocess common docs
        parsed_common_docs = (nlp_model(common_doc.text) for common_doc in common_docs) if common_docs else ()
        for idx, parsed_doc in enumerate(parsed_common_docs):
            self._preprocess(common_docs[idx], parsed_doc.sentences, [], stop_words, join_small_sents=False)
        common_sent_words = [sent.words for doc_sents in (doc.sents(include_common=True) for doc in common_docs)
                             for sent in doc_sents] if common_docs else []
        for idx in trange(len(docs)):
            parsed_doc = nlp_model(docs[idx].text)
            self._preprocess(docs[idx], parsed_doc.sentences, common_sent_words, stop_words)

    def _preprocess(self, doc: Document, sents, common_sent_words: list[list[Word]], stop_words: set[str],
                    join_small_sents=True):
        for sent_idx, sent in enumerate(sents):
            non_punct_words = [word for word in sent.words if not word.upos == 'PUNCT']
            if self._rem_stop_words:
                sent_lemmas = [word.lemma for word in non_punct_words if word.text.lower() not in stop_words]
            else:
                sent_lemmas = [word.lemma for word in non_punct_words]
            if len(sent_lemmas):
                lemma_count = Counter(sent_lemmas)
                sentence = Sentence(sent.tokens[0].start_char, sent.tokens[-1].end_char, lemma_count, doc)
                sentence.words = _to_words(sent.tokens, sentence)
                doc.add_sent(sentence)
                if _sent_contains_words_of_common_sent(sentence.words, common_sent_words):
                    sentence.common = True
                else:
                    for lemma in lemma_count.keys():
                        doc.vocab[lemma] += 1
        if join_small_sents:
            self._join_small_sentences(doc)

    def _join_small_sentences(self, doc: Document):
        sents = doc.sents(include_common=True)
        idx, sent_count = 0, len(sents)
        while idx < sent_count - 1:
            sent1, sent2 = sents[idx], sents[idx + 1]
            if (not sent1.common and not sent2.common) and \
                (sum(sent1.bow.values()) < self._min_sent_len or
                 (sent2 == sents[-1] and sum(sent2.bow.values()) < self._min_sent_len)):
                for lemma in sent1.bow.keys():
                    if lemma in sent2.bow:
                        doc.vocab[lemma] -= 1
                joined_sent = Sentence(sent1.start_char, sent2.end_char, sent1.bow + sent2.bow, doc)
                joined_sent.words = sent1.words + sent2.words
                doc.remove_sent(sent1), doc.remove_sent(sent2)
                doc.add_sent(joined_sent)
                sent_count -= 1
            idx += 1


def _nlp_pipe(lang: str) -> Pipeline:
    if lang in LANG_CODES:
        try:
            return stanza.Pipeline(LANG_CODES[lang], processors=PRCS, logging_level=PIPE_LVL)
        except:  # Unpickling error raises Exception, cannot narrow
            stanza.download(LANG_CODES[lang], processors=PRCS, logging_level=LOAD_LVL)
            return stanza.Pipeline(LANG_CODES[lang], processors=PRCS, logging_level=PIPE_LVL)
    else:
        raise UnsupportedLanguageError(f'The language "{lang}" is currently not supported.')


def _to_words(stanza_tokens, sentence: Sentence) -> list[Word]:
    words = []
    for stanza_token in stanza_tokens:
        for stanza_word in stanza_token.words:
            if not stanza_word.upos == 'PUNCT':
                words.append(Word(stanza_token.start_char, stanza_token.end_char, sentence))
    return words


def _sent_contains_words_of_common_sent(sent_words: list[Word], common_sent_words: list[list[Word]]) -> bool:
    sent_word_texts = [word.text for word in sent_words]
    for sent_words in common_sent_words:
        if all(word.text in sent_word_texts for word in sent_words):
            return True
    return False


class UnsupportedLanguageError(Exception):
    pass
