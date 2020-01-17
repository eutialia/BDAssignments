import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def preprocess_df(self, df, cols):
        df['joinKey'] = df[cols].astype(str).apply(lambda row: ' '.join(row), axis=1)
        df['joinKey'] = df['joinKey'].str.lower().str.replace(' nan', '').str.split(r'\W+')
        return df

    def filtering(self, df1, df2):
        df1['merge'] = df1['joinKey']
        df2['merge'] = df2['joinKey']
        temp = df1.explode('merge').merge(df2.explode('merge'), on='merge', how='inner', suffixes=('1','2'))
        temp = temp.drop_duplicates(['id1', 'id2'])
        cand_df = temp[['id1', 'joinKey1', 'id2', 'joinKey2']].copy()
        return cand_df

    def verification(self, cand_df, threshold):
        jaccard = []
        for joinKey1, joinKey2 in zip(cand_df['joinKey1'], cand_df['joinKey2']):
            jaccard_similarity = len(set(joinKey1) & set(joinKey2)) / len(set(joinKey1) | set(joinKey2)) if len(set(joinKey1) | set(joinKey2)) > 0 or 0
            jaccard.append(jaccard_similarity)
        cand_df['jaccard'] = jaccard
        result_df = cand_df.query(f'jaccard >= {threshold}')
        return result_df

    def evaluate(self, result, ground_truth):
        new_result = [i[0] + i[1] for i in result]
        new_ground_truth = [i[0] + i[1] for i in ground_truth]
        precision = len(set(new_result) & set(new_ground_truth)) / len(new_result)
        recall = len(set(new_result) & set(new_ground_truth)) / len(new_ground_truth)
        fmeasure = (2 * precision * recall) / (precision + recall)
        return (precision, recall, fmeasure)

    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 

        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))

        return result_df



if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))