# NLP_Financial_Sentiment_Analysis
Processing news to get markets sentiment

Finding the financial markets sentiment from textual data (headlines/articles) is the target of the project

Steps and corresponding files:

0) Find the dataset ( available at https://www.kaggle.com/datasets/gennadiyr/us-equities-news-data )

   The dataset includes more than 200k financial news (title and corresponding article) in the timespan 2008-2020 with category (news/opinion),ticker and provider
   
1) Preprocessing --> NLP_FSA_PreProcessing.ipynb
   
   - Cleaning textual data and get financial returns
   - Providing different datasets including news only ones (no opinion articles) 
   
2) CountVectorizer Application --> NLP_FinSA_CountVectorizer.ipynb

   - Binary and various ternary classifications of headlines/articles using financial returns
   - Text vectorization through CountVectorizer
   - Default ML models application
   - Parameters Tuning of Vectorizer and ML models, also searched for scaling benefits 
   
n for n in range(Next_Steps)) Next Steps
   Apply:
   - TfidfVectorizer 
   - Word Embedding (Word2vec)
   - NLP Transformers (FinBERT)



Notes:
- The whole project has been developed on Colab
