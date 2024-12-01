import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

'''Funzione per pulire di dataset, serve per pulire descrizioni'''
def clean_about_product(text):
    text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

'''Manipolazione dataset'''
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 1000000)
df = pd.read_csv('amazon.csv')
df_filtered = df[['product_name', 'product_id', 'about_product', 'category', 'rating']]
df_filtered.to_csv('output.csv', index=False)

df['about_product'] = df['about_product'].apply(clean_about_product)
df_filtered.loc[:, 'about_product'] = df['about_product'].apply(clean_about_product)
df_filtered = df_filtered.drop_duplicates(subset='product_name', keep='first')

'''Vettorizzo le descrizioni e calcolo similarità prodotti'''
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(df_filtered['about_product'])

cosine_sim = cosine_similarity(X_tfidf, X_tfidf)
'''Funzione per ottenere i prodotti più simili ad un certo prodotto in un certo range di rating'''
def get_recommendations_with_rating_range(product_idx, cosine_sim, df_filtered, min_rating=3, max_rating=5, top_n=5):
    df_filtered.loc[:, 'rating'] = pd.to_numeric(df_filtered['rating'], errors='coerce')

    sim_scores = list(enumerate(cosine_sim[product_idx]))

    filtered_sim_scores = [
        (i, score) for i, score in sim_scores if min_rating <= df_filtered['rating'].iloc[i] <= max_rating
    ]

    filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)

    top_products = filtered_sim_scores[:top_n]
    product_indices = [x[0] for x in top_products]

    recommendations = []
    for idx in product_indices:
        product_name = df_filtered['product_name'].iloc[idx]
        product_rating = df_filtered['rating'].iloc[idx]
        recommendations.append((product_name, product_rating))

    return recommendations


'''Test e print'''
recommended_products_with_rating_range = get_recommendations_with_rating_range(167, cosine_sim, df_filtered)

for product, rating in recommended_products_with_rating_range:
    print(f"Product: {product}, Rating: {rating}")
