import pandas as pd
import re
from text_tools import TextTools

from NLPInfrastructure.normalizer import SentenceNormalizer

from NLPInfrastructure.resources import stopWords, prepositions, postWords, Not1Gram

normalizer = SentenceNormalizer()


def remove_stopword(text):
    text = text.replace('.', ' ').replace(',', ' ')
    words = text.split(' ')
    words_filtered = []
    for w in words:
        if (w not in stopWords) and (w not in prepositions) and (w not in postWords) and (w not in Not1Gram):
            words_filtered.append(w)

    res = ' '.join(words_filtered)
    res = res.strip()
    return res


def normalize_text_atomic(text):
    text = normalizer.organize_text(text)
    text = normalizer.replace_urls(text, '')
    text = normalizer.replace_emails(text, '')
    text = normalizer.replace_usernames(text)
    # text = normalizer.replace_hashtags(text, 'MyHashtag')
    text = normalizer.edit_arabic_letters(text)
    text = normalizer.replace_phone_numbers(text)
    text = normalizer.replace_emoji(text)
    text = normalizer.replace_duplicate_punctuation(text)

    text = normalizer.replace_consecutive_spaces(text)
    text = remove_stopword(text)

    return text


def normalize_texts(texts):
    return [normalize_text_atomic(text) for text in texts]


if __name__ == '__main__':
    tt = TextTools()
    texts = pd.read_csv('data/sample_fa.csv')

    texts = texts[texts['target'] == 1]

    texts['id'] = texts['id'].astype('int64').astype(str)

    # texts = texts.sample(frac=0.1)

    texts['clean_text'] = normalize_texts(texts['text'])

    allkeywords = tt.text_to_allkeywords(texts)
    graph = tt.create_graph(allkeywords)
    graph.to_csv(r'data/graph_sample_fa.csv', index=False)
    # graph = pd.read_csv('data/sample_fa_graph.csv')

    print('done')
