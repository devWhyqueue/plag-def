from collections import Counter

import pytest

from plagdef.model.pipeline.preprocessing import UnsupportedLanguageError, Document, Preprocessor, _nlp_pipe, \
    _extract_urls


def test_nlp_model():
    en = _nlp_pipe('eng')
    de = _nlp_pipe('ger')
    assert en.lang == 'en'
    assert de.lang == 'de'


def test_nlp_model_with_wrong_lang():
    with pytest.raises(UnsupportedLanguageError):
        _nlp_pipe('fre')


def test_ger_sent_seg(nlp_ger):
    doc = nlp_ger('Das ist ein schöner deutscher Text. Besteht aus zwei Sätzen.')
    sent1_words, sent2_words = doc.sentences[0].words, doc.sentences[1].words
    assert len(doc.sentences) == 2
    assert sent1_words[0].text == 'Das' and sent2_words[0].text == 'Besteht'


def test_ger_sent_seg_with_same_sent_starts(nlp_ger):
    doc = nlp_ger('Das ist ein schöner deutscher Text. Das ist auch einer.')
    sent1_tokens, sent2_tokens = doc.sentences[0].tokens, doc.sentences[1].tokens
    assert len(doc.sentences) == 2
    assert sent1_tokens[0].text == 'Das' and sent2_tokens[0].text == 'Das'
    assert sent1_tokens[0].start_char != sent2_tokens[0].start_char


def test_ger_sent_seg_with_next_sent_start_in_previous_sent(nlp_ger):
    doc = nlp_ger('Eine Klasse besteht aus Methoden und Attributen. Methoden realisieren Verhalten.')
    sent1_tokens, sent2_tokens = doc.sentences[0].tokens, doc.sentences[1].tokens
    assert len(doc.sentences) == 2
    assert sent1_tokens[0].text == 'Eine' and sent2_tokens[0].text == 'Methoden'
    assert sent1_tokens[0].start_char == 0 and sent2_tokens[0].start_char == 49


def test_ger_sent_seg_with_paragraphs(nlp_ger):
    doc = nlp_ger('Ein schöner deutscher Text\n\nKurzer Satz.\nUnd noch ein etwas längerer.')
    sent1_words, sent2_words, sent3_words = doc.sentences[0].words, doc.sentences[1].words, doc.sentences[2].words
    assert len(doc.sentences) == 3
    assert sent1_words[0].text == 'Ein' and sent2_words[0].text == 'Kurzer' and sent3_words[0].text == 'Und'


def test_preprocessed_voc_contains_lemmas_with_sentence_frequency(preprocessed_docs):
    doc = preprocessed_docs[0]
    # One error (ignored): rights -> right
    assert doc.vocab == Counter(
        {'be': 3, 'infringement': 2, 'the': 2, 'copyright': 2, 'plagiarism': 1, 'not': 1, 'same': 1, 'as': 1,
         'to': 1, 'concept': 1, 'apply': 1, 'they': 1, 'may': 1, 'while': 1, 'different': 1, 'act': 1, 'particular': 1,
         'term': 1, 'both': 1, 'material': 1, 'consent': 1, 'holder': 1, 'whose': 1, 'violation': 1, 'of': 1, 'use': 1,
         'without': 1, 'restrict': 1, 'when': 1, 'rights': 1, 'by': 1})


def test_tokenize_voc_contains_german_lemmas(preprocessor):
    doc = Document('doc', 'path/to/doc',
                   'Als Wortstamm oder kurz Stamm bezeichnet man in der Grammatik '
                   'einen Bestandteil eines Wortes, der als Ausgangsbasis für '
                   'weitere Wortbildung dienen kann. Es handelt sich demnach um '
                   'ein potenziell unvollständiges Gebilde, das als Gegenstück zu '
                   'einem Affix auftreten kann.')
    preprocessor.preprocess('ger', [doc])
    # Errors (ignored):
    # Term 'sich' is mapped to 'er|sie|es' but not 'es'
    # Either map all (reflexive) pronouns to 'er|es|sie' or none
    assert list(doc.vocab.keys()) \
           == ['als', 'Wortstamm', 'oder', 'kurz', 'Stamm', 'bezeichnen', 'man',
               'in', 'der', 'Grammatik', 'ein', 'Bestandteil', 'Wort',
               'Ausgangsbasis', 'für', 'weit', 'Wortbildung', 'dienen', 'können',
               'es', 'handeln', 'er|es|sie', 'demnach', 'um', 'potenziell',
               'unvollständig', 'Gebilde', 'Gegenstück', 'zu', 'Affix',
               'auftreten']


def test_preprocessed_voc_contains_no_stop_words():
    preprocessor = Preprocessor(3, True)
    doc1 = Document('doc1', 'path/to/doc1', 'This is a sentence with five stop words.')
    doc2 = Document('doc2', 'path/to/doc2', 'Another document for good measure.')
    preprocessor.preprocess('eng', [doc1, doc2])
    assert len(doc1.vocab.keys()) == 3


def test_preprocessed_voc_contains_no_short_tokens(preprocessed_docs):
    doc = preprocessed_docs[0]
    assert not any(len(token) < 2 for token in doc.vocab)


def test_preprocessed_voc_contains_only_lowercase_tokens(preprocessed_docs):
    doc = preprocessed_docs[0]
    assert any(token.islower() for token in doc.vocab)


def test_preprocessed_words(preprocessor):
    doc = Document('doc', 'path/to/doc', 'This is a document. It consists of two sentences: one and two.')
    preprocessor.preprocess('eng', [doc])
    sent1_word_texts = [word.text.lower() for word in doc.sents(include_common=True)[0].words]
    sent2_word_texts = [word.text.lower() for word in doc.sents(include_common=True)[1].words]
    assert sent1_word_texts == ['this', 'is', 'document']
    assert sent2_word_texts == ['it', 'consists', 'of', 'two', 'sentences', 'one', 'and', 'two']


def test_preprocessed_words_are_alphanumeric(preprocessor):
    doc = Document('doc', 'path/to/doc', 'This is a n3xt l€v€l document.')
    preprocessor.preprocess('eng', [doc])
    sent_word_texts = [word.text.lower() for word in doc.sents(include_common=True)[0].words]
    assert sent_word_texts == ['this', 'is', 'n3xt', 'document']


def test_preprocessed_sent_start_end_chars(preprocessed_docs):
    doc = preprocessed_docs[1]
    assert [(sent.start_char, sent.end_char) for sent in doc.sents(include_common=True)] == [(0, 177), (178, 231),
                                                                                             (232, 331)]


def test_preprocessed_sent_start_end_chars_after_join(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'Short sentence. Short sentence should be joined with this one.')
    doc2 = Document('doc2', 'path/to/doc2', 'Another document for good measure.')
    preprocessor.preprocess('eng', [doc1, doc2])
    # First sentence: (0, 15)
    # Second sentence: (16, 46)
    # Combined: (0, 62) because of whitespace gap between sents
    assert len(doc1.sents(include_common=True)) == 1
    assert (doc1.sents(include_common=True)[0].start_char, doc1.sents(include_common=True)[0].end_char) == (0, 62)


def test_preprocessed_sent_bows(preprocessed_docs):
    doc = preprocessed_docs[0]
    # Lemma error "rights" ignored
    assert [sent.bow for sent in doc.sents(include_common=True)] == [
        Counter({'plagiarism': 1, 'be': 1, 'not': 1, 'the': 1, 'same': 1, 'as': 1, 'copyright': 1, 'infringement': 1}),
        Counter({'while': 1, 'both': 1, 'term': 1, 'may': 1, 'apply': 1, 'to': 1, 'particular': 1, 'act': 1,
                 'they': 1, 'be': 1, 'different': 1, 'concept': 1}),
        Counter({'copyright': 3, 'be': 3, 'of': 2, 'use': 2, 'infringement': 1, 'violation': 1, 'the': 1,
                 'rights': 1, 'holder': 1, 'when': 1, 'material': 1, 'whose': 1, 'restrict': 1, 'by': 1, 'without': 1,
                 'consent': 1})]


def test_preprocessed_not_any_of_sent_bows_empty(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'Das ist ein schöner Satz.\n\nNoch ein schöner Satz.')
    doc2 = Document('doc2', 'path/to/doc2', 'Another document for good measure.')
    preprocessor.preprocess('ger', [doc1, doc2])
    assert len([sent.bow for sent in doc1.sents(include_common=True)]) == 2
    assert any([len(sent.bow) for sent in doc1.sents(include_common=True)])


def test_preprocessed_bows_contains_only_lowercase_tokens(preprocessed_docs):
    doc = preprocessed_docs[0]
    for sent in doc.sents(include_common=True):
        bow = sent.bow
        assert any([token.islower() for token in bow])


def test_join_small_sents_at_start(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'Short sentence. Short sentence should be joined with this one.')
    doc2 = Document('doc2', 'path/to/doc2', 'Another document for good measure.')
    preprocessor.preprocess('eng', [doc1, doc2])
    assert len(doc1.sents(include_common=True)) == 1
    assert [sent.bow for sent in doc1.sents(include_common=True)] == \
           [Counter({'short': 2, 'sentence': 2, 'should': 1, 'be': 1, 'join': 1, 'with': 1, 'this': 1, 'one': 1})]
    assert doc1.vocab == Counter(
        {'sentence': 1, 'short': 1, 'this': 1, 'should': 1, 'one': 1, 'be': 1, 'with': 1, 'join': 1})


def test_join_multiple_small_sents(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'Short. Still. But this is a longer one.')
    doc2 = Document('doc2', 'path/to/doc2', 'Another document for good measure.')
    preprocessor.preprocess('eng', [doc1, doc2])
    assert len(doc1.sents(include_common=True)) == 1
    assert [sent.bow for sent in doc1.sents(include_common=True)] == \
           [Counter({'short': 1, 'still': 1, 'but': 1, 'this': 1, 'be': 1, 'long': 1, 'one': 1})]
    assert [word.text.lower() for word in doc1.sents(include_common=True)[0].words] \
           == ['short', 'still', 'but', 'this', 'is', 'longer', 'one']
    assert doc1.vocab == Counter(
        {'short': 1, 'still': 1, 'but': 1, 'this': 1, 'be': 1, 'long': 1, 'one': 1})


def test_join_small_sents_in_the_middle(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'This is a long sentence. Short sentence. Short sentence should be '
                                            'joined with this one.')
    doc2 = Document('doc2', 'path/to/doc2', 'Another document for good measure.')
    preprocessor.preprocess('eng', [doc1, doc2])
    assert len(doc1.sents(include_common=True)) == 2
    assert [sent.bow for sent in doc1.sents(include_common=True)] == \
           [Counter({'this': 1, 'be': 1, 'long': 1, 'sentence': 1}),
            Counter({'short': 2, 'sentence': 2, 'should': 1, 'be': 1, 'join': 1, 'with': 1, 'this': 1, 'one': 1})]
    assert doc1.vocab == Counter(
        {'be': 2, 'sentence': 2, 'this': 2, 'long': 1, 'short': 1, 'one': 1, 'join': 1, 'with': 1, 'should': 1})


def test_join_small_sents_at_the_end(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'This is a long sentence. Short sentence should be '
                                            'joined with this one. Short sentence.')
    doc2 = Document('doc2', 'path/to/doc2', 'Another document for good measure.')
    preprocessor.preprocess('eng', [doc1, doc2])
    assert len(doc1.sents(include_common=True)) == 2
    assert [sent.bow for sent in doc1.sents(include_common=True)] == \
           [Counter({'this': 1, 'be': 1, 'long': 1, 'sentence': 1}),
            Counter({'short': 2, 'sentence': 2, 'should': 1, 'be': 1, 'join': 1, 'with': 1, 'this': 1, 'one': 1})]
    assert doc1.vocab == Counter(
        {'be': 2, 'sentence': 2, 'this': 2, 'long': 1, 'short': 1, 'one': 1, 'join': 1, 'with': 1, 'should': 1})


def test_join_small_sents_contains_words_of_both_sents(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'Short sentence. Short sentence should be joined with this one.')
    doc2 = Document('doc2', 'path/to/doc2', 'Another document for good measure.')
    preprocessor.preprocess('eng', [doc1, doc2])
    assert len(doc1.sents(include_common=True)) == 1
    assert [word.text for word in doc1.sents(include_common=True)[0].words] == \
           ['Short', 'sentence', 'Short', 'sentence', 'should', 'be', 'joined', 'with', 'this', 'one']


def test_join_small_sents_does_not_join_common_sent(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'This is the first document. Common sent.')
    doc2 = Document('doc2', 'path/to/doc2', 'This sentence could be part of doc1 but is not.\nCommon sent.')
    preprocessor.preprocess('eng', [doc1], [doc2])
    assert len(doc1.sents(include_common=True)) == 2
    assert doc1.sents(include_common=True)[1].common
    assert doc1.vocab == Counter({'this': 1, 'be': 1, 'the': 1, 'first': 1, 'document': 1})


def test_remove_small_sents_near_common_sents(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'This is the first document. Common sent. Remove this.')
    doc2 = Document('doc2', 'path/to/doc2', 'This sentence could be part of doc1 but is not.\nCommon sent.')
    preprocessor.preprocess('eng', [doc1], [doc2])
    assert len(doc1.sents(include_common=True)) == 2
    assert doc1.sents(include_common=True)[1].common
    assert doc1.vocab == Counter({'this': 1, 'be': 1, 'the': 1, 'first': 1, 'document': 1})


def test_tag_common_sents(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1',
                    'This is the first document. Last sentence is expected to be common to all docs.')
    doc2 = Document('doc2', 'path/to/doc2', 'This sentence could be part of doc1 but is not.\n'
                                            'Last Sentence, is expected to be common to all docs.')
    preprocessor.preprocess('eng', [doc1], [doc2])
    assert len(doc1.sents(include_common=True)) == 2
    assert doc1.sents(include_common=True)[1].common
    assert doc1.vocab == Counter({'this': 1, 'be': 1, 'the': 1, 'first': 1, 'document': 1})


def test_tag_common_sents_with_empty_line(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1',
                    'This is the first document. Last sentence is expected to be common to all docs.')
    doc2 = Document('doc2', 'path/to/doc2', 'This sentence could be part of doc1 but is not.\n'
                                            '\n'
                                            'Last Sentence, is expected to be common to all docs.')
    preprocessor.preprocess('eng', [doc1], [doc2])
    assert len(doc1.sents(include_common=True)) == 2
    assert len(list(doc1.sents())) == 1
    assert doc1.sents(include_common=True)[1].common
    assert doc1.vocab == Counter({'this': 1, 'be': 1, 'the': 1, 'first': 1, 'document': 1})


def test_tag_common_sents_with_words(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1',
                    'This is the first document. Last sentence is expected to be common to all docs.')
    doc2 = Document('doc2', 'path/to/doc2', 'Some. random. words\nand something more. relevant:\ncommon to be expected')
    preprocessor.preprocess('eng', [doc1], [doc2])
    assert len(doc1.sents(include_common=True)) == 2
    assert doc1.sents(include_common=True)[1].common
    assert doc1.vocab == Counter({'this': 1, 'be': 1, 'the': 1, 'first': 1, 'document': 1})


def test_tag_common_sents_with_non_identical_sents(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1', 'This is the first document.\n'
                                            'Presenting the last sentence:\n'
                                            'Last sentence is expected to be common to all docs.')
    doc2 = Document('doc2', 'path/to/doc2', 'This sentence could be part of doc1 but is not.\n'
                                            'Last sentence is expected to be common to all docs.')
    preprocessor.preprocess('eng', [doc1], [doc2])
    assert len(doc1.sents(include_common=True)) == 2
    assert doc1.sents(include_common=True)[1].common
    assert doc1.vocab == Counter({'this': 1, 'be': 1, 'the': 1, 'first': 1, 'document': 1})


def test_extract_urls():
    doc = Document("doc", "path/to/doc", "Google.com is the most popular search engine, www.ecosia.com is more "
                                         "ecologically friendly and https://bing.com is the default engine of "
                                         "Microsoft Edge.")
    _extract_urls(doc)
    assert doc.urls == {"https://Google.com", "https://www.ecosia.com", "https://bing.com"}


def test_extract_urls_if_urls_present(preprocessor):
    doc1 = Document("doc1", "path/to/doc", "This is an URL: www.google.de/search?q=python")
    doc1.urls = {"https://www.bing.de"}
    doc2 = Document('doc2', 'path/to/doc2', 'Another document for good measure.')
    preprocessor.preprocess('eng', [doc1], [doc2])
    assert doc1.urls == {"https://www.bing.de", "https://www.google.de/search?q=python"}


def test_extract_urls_with_subpath():
    doc = Document("doc", "path/to/doc", "This is an URL: www.google.de/search?q=python")
    _extract_urls(doc)
    assert doc.urls == {"https://www.google.de/search?q=python"}


def test_extract_urls_filters_duplicates():
    doc = Document("doc", "path/to/doc", "These URLs are both the same: https://google.de, google.de")
    _extract_urls(doc)
    assert doc.urls == {"https://google.de"}


def test_extract_url_normalizes_address():
    doc = Document("doc", "path/to/doc", "Some URLs: google.de, https://google.de/abc and google.de/abc/")
    _extract_urls(doc)
    assert doc.urls == {'https://google.de/abc', 'https://google.de'}


def test_extract_url_rstrips_punctuation():
    doc = Document("doc", "path/to/doc",
                   "https://www.weltderphysik.de/gebiet/teilchen/antimaterie/antimaterie-im-universum/,")
    _extract_urls(doc)
    assert doc.urls == {'https://www.weltderphysik.de/gebiet/teilchen/antimaterie/antimaterie-im-universum'}
