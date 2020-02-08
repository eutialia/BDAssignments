import pandas as pd
from sklearn.cluster import KMeans

class AnomalyDetection():
    def __init__(self):
        super().__init__()

    def scaleNum(self, df, indices):
        for i in indices:
            dominating_feature = pd.Series([data[i] for data in df['features']])
            mean = dominating_feature.mean()
            std = dominating_feature.std()
            standarized_feature = dominating_feature.apply(lambda x: (x-mean)/std)
            for j in range(df['features'].size):
                df['features'][j][i] = standarized_feature[j]
        return df

    def cat2Num(self, df, indices):
        tokens = ['http', 'ftp', 'udt', 'udf', 'tcp', 'icmp'] # same sequence as example for testing purpose
        # tokens = list(filter(lambda x: type(x) is str, df.explode('features')['features'].unique()))
        encoder = dict((keys, values) for values, keys in enumerate(tokens))
        df['features'] = df['features'].apply(self.encode, args=(indices, encoder,)).apply(self.insert, args=(indices, tokens,))
        return df

    def detect(self, df, k, t):
        df['score'] = KMeans(n_clusters=k, random_state=0).fit_predict(df['features'].to_list())
        n_max = df.groupby('score', as_index=False).count().max().features
        n_min = df.groupby('score', as_index=False).count().min().features
        df['score'] = df['score'].apply(lambda x: (n_max - x) / (n_max - n_min))
        return df.query('score >= @t')

    def encode(self, data, indices, encoder):
        for i in indices:
            data[i] = encoder[data[i]]
        return data
    
    def insert(self, data, indices, tokens):
        base = [0 for _ in range(len(tokens))]
        for i in reversed(indices):
            base[data[i]] = 1
            data.pop(i)
            for j in reversed(base):
                if i == indices[0]:
                    data.insert(i, j)
        return data

if __name__ == "__main__":
    # df = pd.read_csv('logs-features-sample.csv', converters={'features': eval}).set_index('id')
    data = [(0, ["http", "udt", 4]), \
            (1, ["http", "udf", 5]), \
            (2, ["http", "tcp", 5]), \
            (3, ["ftp", "icmp", 1]), \
            (4, ["http", "tcp", 4])]
    df = pd.DataFrame(data=data, columns = ["id", "features"])
    ad = AnomalyDetection()

    df1 = ad.cat2Num(df, [0,1])
    print(df1)

    df2 = ad.scaleNum(df1, [6])
    print(df2)

    df3 = ad.detect(df2, 2, 0.9)
    print(df3)
