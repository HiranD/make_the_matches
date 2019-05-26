from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from tqdm import tqdm
import multiprocessing  
from multiprocessing import Pool

df = pd.read_csv("dataset_cleaned.csv")
docs = [(row['text'], row['DALTIX_ID']) for index, row in df.iterrows()]
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85).fit_transform(df.text.tolist())

def cal_function(i):
    cosine_similarities = linear_kernel(tfidf[i], tfidf).flatten()
    related_docs_indices = (-cosine_similarities).argsort()[:2]
    for n in related_docs_indices:
        if n == i:
            continue
        else:
            return(df['DALTIX_ID'].iat[i], df['DALTIX_ID'].iat[n], cosine_similarities[n])
            break

if __name__ == '__main__':
    print("processing..")
    with Pool(8) as p:
          result = list(tqdm(p.imap(cal_function, range(len(docs))), total=len(docs)))
            
    print("saving..")
    pd.DataFrame(result, columns=['daltix_id_1', 'daltix_id_2', 'similarity']).to_csv("tfidf_df_ngrams.csv", index=False)