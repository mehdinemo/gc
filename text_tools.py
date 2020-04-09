import pandas as pd
import numpy as np


class TextTools:
    def _text_to_allkeywords(self, data: pd.DataFrame) -> pd.DataFrame:
        data['transactions'] = data['clean_text'].apply(lambda t: list(filter(None, t.split(' '))))

        allkeywords = pd.DataFrame({'message_id': np.repeat(data['id'].values, data['transactions'].str.len()),
                                    'word': np.concatenate(data['transactions'].values)})
        allkeywords['count'] = 1
        allkeywords = allkeywords.groupby(['message_id', 'word'], as_index=False).sum()
        allkeywords = allkeywords.merge(data[['id', 'target']], how='left', left_on='message_id', right_on='id')
        allkeywords.drop(['id'], axis=1, inplace=True)
        return allkeywords

    def _create_graph(self, allkeywords):
        graph = allkeywords.merge(allkeywords, how='inner', left_on='word', right_on='word')
        graph = graph.groupby(['message_id_x', 'message_id_y'], as_index=False)['word'].count()

        graph.columns = ['source', 'target', 'weight']
        return graph
