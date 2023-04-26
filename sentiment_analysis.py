import pandas as pd
from textblob import TextBlob
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

def preprocess_review(review):
    review = review.lower()
    return TextBlob(review)

def sentiment_score(review):
    sentence_scores = [sentence.sentiment.polarity for sentence in review.sentences]
    overall_score = sum(sentence_scores) / len(sentence_scores)
    return overall_score

def sentiment_label(score):
    return 'positive' if score > 0 else 'negative'

def extract_collocations(reviews, pos_filter=None):
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents(reviews)

    if pos_filter:
        finder.apply_ngram_filter(lambda w1, w2: (w1[1], w2[1]) != pos_filter)

    return finder.nbest(bigram_measures.pmi, 40)

# Load the dataset
data = pd.read_csv('movie_reviews.csv', delimiter='\t', header=None, names=['review_id', 'movie_id', 'review'])

# Preprocess the reviews
data['review'] = data['review'].apply(preprocess_review)

# Perform sentiment analysis using TextBlob
data['sentiment_score'] = data['review'].apply(sentiment_score)
data['sentiment_label'] = data['sentiment_score'].apply(sentiment_label)

# Extract collocations from positive and negative reviews
positive_reviews = data[data['sentiment_label'] == 'positive']['review'].tolist()
negative_reviews = data[data['sentiment_label'] == 'negative']['review'].tolist()

positive_collocations = extract_collocations(positive_reviews)
negative_collocations = extract_collocations(negative_reviews)

# Compare collocation extraction with and without part-of-speech filtering
positive_collocations_filtered = extract_collocations(positive_reviews, ('JJ', 'NN'))
negative_collocations_filtered = extract_collocations(negative_reviews, ('JJ', 'NN'))

print("Positive collocations without filtering:")
print(positive_collocations)
print("Positive collocations with POS filtering:")
print(positive_collocations_filtered)

print("Negative collocations without filtering:")
print(negative_collocations)
print("Negative collocations with POS filtering:")
print(negative_collocations_filtered)
