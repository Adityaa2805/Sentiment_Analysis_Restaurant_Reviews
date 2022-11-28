import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Loading MNB Classifier and CountVectorizer instance
classifier = pickle.load(open('reviews_classifier.sav', 'rb'))
cv = pickle.load(open('count_vectorizer.sav', 'rb'))

# Defining stop words
stop_words = set(stopwords.words('english'))
negative_words = ['no', 'not', 'none', 'nothing', 'never', 'neither', 'nor']
stop_words = stop_words - set(negative_words)


# Function to predict user sentiments based on input review
def predict_sentiment(input_review):

    input_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=input_review)
    input_review = input_review.lower()
    input_review_words = input_review.split()
    input_review_words = [
        word for word in input_review_words if not word in stop_words]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in input_review_words]
    final_review = ' '.join(final_review)

    temp = cv.transform([final_review]).toarray()
    return classifier.predict(temp)


def main():
    print("\n-------------------------------------------------------------------------------------")
    user_review = input("Enter the review of the restaurant:  ")
    predicition = predict_sentiment(user_review)
    print("-------------------------------------------------------------------------------------")

    if (predicition == 0):
        print("The review entered was negative.")
        print("The user did not like the restaurant.")
    else:
        print("The review entered was positive.")
        print("The user liked the restaurant.")


if __name__ == "__main__":
    main()
