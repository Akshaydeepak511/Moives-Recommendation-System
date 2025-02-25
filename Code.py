import pandas as pd
import numpy as np
df=pd.read_csv(r"Movies Recommendation.csv")
print(df)
df.head()
df.info()
df.shape
df.columns
df_features=df[['Movie_Genre','Movie_Keywords','Movie_Tagline','Movie_Cast','Movie_Director']].fillna('')
df_features.shape
df_features
x=df_features['Movie_Genre']+' '+df_features['Movie_Keywords']+' '+df_features['Movie_Tagline']+' '+df_features['Movie_Cast']+' '+df_features['Movie_Director']
print(x)
x.shape
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
x=tfidf.fit_transform(x)
x.shape
print(x)
from sklearn.metrics.pairwise import cosine_similarity
Similarity_Score=cosine_similarity(x)
Similarity_Score
Similarity_Score.shape
Favourite_Movie_Name=input('Enter your favourite movie name
All_Movies_Title_List=df['Movie_Title'].tolist()
import difflib
Movie_Recommendation=difflib.get_close_matches(Favourite_Movie_Name,All_Movies_Title_List)
print(Movie_Recommendation)
Close_Match=Movie_Recommendation[0]
print(Close_Match)
Index_of_Close_Match_Movie=df[df.Movie_Title==Close_Match]['Movie_ID'].values[0]
print(Index_of_Close_Match_Movie)
#getting a list of similar movies
Recommendation_Score=list(enumerate(Similarity_Score[Index_of_Close_Match_Movie]))
print(Recommendation_Score)
len(Recommendation_Score)
#sorting the movies based on their similarity score
Sorted_Similar_Movies=sorted(Recommendation_Score,key=lambda x:x[1],reverse=True)
print(Sorted_Similar_Movies)
#print the name of similar movies based on the index
print('Top 30 movies suggested for you : \n')
i=1
for movie in Sorted_Similar_Movies:
  index=movie[0]
  title_from_index=df[df.index==index]['Movie_Title'].values[0]
  if(i<31):
    print(i,'.',title_from_index)
    i+=1
#Top 10 movie recommendation system
Movie_Name=input('Enter your favourite movie name:')
list_of_all_titles=df['Movie_Title'].tolist()
Find_Close_Match=difflib.get_close_matches(Movie_Name,list_of_all_titles)
Close_Match=Find_Close_Match[0]
Index_of_Movie=df[df.Movie_Title==Close_Match]['Movie_ID'].values[0]
Recommendation_Score=list(enumerate(Similarity_Score[Index_of_Movie]))
sorted_similar_movies=sorted(Recommendation_Score,key=lambda x:x[1],reverse=True)
print('Top 10 Movies suggested for you: \n')
i=1
for movie in sorted_similar_movies:
  index=movie[0]
  title_from_index=df[df.Movie_ID==index]['Movie_Title'].values
  if(i<11):
    print(i,'.',title_from_index)
    i+=1


























