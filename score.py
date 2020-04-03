import pandas as pd
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()


def main():
    eig_deg = pd.read_csv('data/degrees_df_eig.csv')
    weight_deg = pd.read_csv('data/degrees_df_deg.csv')

    eig_deg.set_index('node', inplace=True)
    weight_deg.set_index('node', inplace=True)

    degree_df = eig_deg.merge(weight_deg, how='inner', left_index=True, right_index=True)
    degree_df.drop(['class_x'], axis=1, inplace=True)

    degree_df.columns = ['degree_eig', 'class_degree_eig', 'degree_we', 'class', 'class_degree_we']

    degree_df['degree_eig'] = degree_df['degree_eig'] / degree_df['degree_eig'].sum()
    # degree_df['degree_eig'] = min_max_scaler.fit_transform(degree_df['degree_eig'].values.reshape(-1, 1))
    classes = degree_df['class'].unique()
    for c in classes:
        tmp = degree_df[degree_df['class'] == c]['class_degree_eig']

        t = tmp / tmp.sum()

        # t = tmp.values.reshape(-1, 1)
        # t = min_max_scaler.fit_transform(t)

        degree_df.loc[degree_df['class'] == c, 'class_degree_eig'] = t

    degree_df['score'] = degree_df['class_degree_eig'] - degree_df['degree_eig']

    degree_df.to_csv('data/score_df.csv')
    print('done')


if __name__ == '__main__':
    main()
