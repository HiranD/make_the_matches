# Make the matches!

## Approches

* Gensim doc2vec
* sk-learn tf-idf
* Spacy Document Similarity

### Gensim doc2vec

Results were not good so moved to tf-idf.. But I think we should be able to tune the procedure for better results.

### sk-learn tf-idf 

Got a f1 score of 0.60 using the given validation data.

### Spacy Document Similarity (using nl_core_news_sm model)

Just done some experiments. Seems like tf-idf gave slightly better results. Havent had enough time to continue.

## Code

Removed data folder from this repo because of github limitations.
This repo contains several Jupyter Notebooks and data sets saved at checkpoints to make it easy to do changes from a checkpoint.

* preprocess_data.ipynb - Preprocess data and save dataframe as dataset_cleaned.csv
* doc2vec.ipynb - Find simalarity scores between produts using gensim doc2vec and save results as doc2vec_df.csv
* tfidf.ipynb - Find simalarity scores between produts using sk-learn tf-idf and save results as tfidf_df_ngrams.csv
* tf_idf.py - All the functionalities in the tfidf.ipynb. Because had problems of running parallel processes in Jupyter.
* validate_results.ipynb - Validate results using given dataset and given f1 score.
* testings.ipynb - some testings and EDA on tf-idf results.

Some parts of the codes take hours to process. It can be solved if I have used Dask Dataframes. Havent had enough time to do that.

### Max f1 score achived - 0.60056

## Instructions to produce same results

* Clone the repo and copy the original data folder to it.
* Firat run preprocess_data.ipynb and produce dataset_cleaned.csv.
* Then run the tf_idf.py and produce tfidf_df_ngrams.csv (This process take time, therefor i have commited the tfidf_df_ngrams.csv).
* Finally run validate_results.ipynb and see the f1 Score.
