import pandas as pd

if __name__ == "__main__":
    dataset = pd.read_csv("./revlog_history.tsv", sep='\t', index_col=None, dtype={'r_history': str ,'t_history': str} )
    dataset['r_history'].fillna('', inplace=True)
    dataset['t_history'].fillna('', inplace=True)
    dataset.review_time = dataset.review_time.astype(int)
    dataset.card_id = dataset.card_id.astype(int)
    dataset.review_duration = dataset.review_duration.astype(int)
    dataset.review_state = dataset.review_state.astype(int)
    dataset.review_rating = dataset.review_rating.astype(int)
    dataset.delta_t = dataset.delta_t.astype(int).astype(str)
    dataset.y = dataset.y.astype(int).astype(str)
    dataset.i = dataset.i.astype(int)
    delta_ts = dataset.groupby('card_id')['delta_t'].apply(lambda x: ','.join(x.values[1:])).to_dict()
    ys = dataset.groupby('card_id')['y'].apply(lambda x: ','.join(x.values[1:])).to_dict()
    dataset.drop_duplicates(subset=['card_id'], keep='last', inplace=True)
    dataset = dataset[(dataset['i'] >= 2)].copy()
    dataset['delta_ts'] = dataset['card_id'].map(delta_ts)
    dataset['ys'] = dataset['card_id'].map(ys)
    dataset[['t_history', 'r_history', 'delta_ts', 'ys']].to_csv("./seq2seq.tsv", sep='\t', index=False)