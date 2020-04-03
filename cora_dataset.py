import pandas as pd
import time


def prepare_data():
    cites = pd.read_csv(r'D:\git\data\cora\cites.csv', sep='\t', header=None)
    cites.columns = ['cited', 'citing']

    content = pd.read_csv(r'D:\git\data\cora\content.csv', sep='\t', header=None)
    # content.set_index(0, inplace=True)
    content.rename(columns={0: 'paper_id', content.columns[-1]: 'classes'}, inplace=True)
    classes = content[['paper_id', 'classes']].copy()
    content.drop(['classes'], axis=1, inplace=True)

    allkeywords = content.melt('paper_id', var_name='word', value_name='count')
    allkeywords = allkeywords[allkeywords['count'] != 0]
    allkeywords.drop(['count'], axis=1, inplace=True)

    graph = allkeywords.merge(allkeywords, how='inner', left_on='word', right_on='word')

    start_time = time.time()
    graph = graph.groupby(['paper_id_x', 'paper_id_y'], as_index=False)['word'].sum()
    end_time = time.time()

    print(end_time - start_time)

    graph.columns = ['source', 'target', 'weight']

    graph.to_csv('cora_graph.csv', index=False)

    return cites


def main():
    prepare_data()
    print('done')


if __name__ == '__main__':
    main()
