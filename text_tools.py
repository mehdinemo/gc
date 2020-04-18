import pandas as pd
import numpy as np
from NLPInfrastructure.normalizer import SentenceNormalizer

from NLPInfrastructure.resources import stopWords, prepositions, postWords, Not1Gram

normalizer = SentenceNormalizer()


class TextTools:
    def text_to_allkeywords(self, data: pd.DataFrame) -> pd.DataFrame:
        data['transactions'] = data['clean_text'].apply(lambda t: list(filter(None, t.split(' '))))

        allkeywords = pd.DataFrame({'message_id': np.repeat(data['id'].values, data['transactions'].str.len()),
                                    'word': np.concatenate(data['transactions'].values)})
        allkeywords['count'] = 1
        allkeywords = allkeywords.groupby(['message_id', 'word'], as_index=False).sum()
        # allkeywords = allkeywords.merge(data[['id', 'target']], how='left', left_on='message_id', right_on='id')
        # allkeywords.drop(['id'], axis=1, inplace=True)
        return allkeywords

    def create_graph(self, allkeywords):
        graph = allkeywords.merge(allkeywords, how='inner', left_on='word', right_on='word')
        graph = graph.groupby(['message_id_x', 'message_id_y'], as_index=False)['word'].count()

        graph.columns = ['source', 'target', 'weight']
        graph.drop(graph.loc[graph['source'] > graph['target']].index.tolist(), inplace=True)
        return graph

    def _remove_stopword(self, text):
        text = text.replace('.', ' ').replace(',', ' ')
        words = text.split(' ')
        words_filtered = []
        for w in words:
            if (w not in stopWords) and (w not in prepositions) and (w not in postWords) and (w not in Not1Gram):
                words_filtered.append(w)

        res = ' '.join(words_filtered)
        res = res.strip()
        return res

    def _normalize_text_atomic(self, text):
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
        text = self._remove_stopword(text)

        return text

    def normalize_texts(self, texts):
        return [self._normalize_text_atomic(text) for text in texts]
