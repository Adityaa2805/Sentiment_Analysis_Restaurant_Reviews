import pickle
from sklearn.feature_extraction.text import CountVectorizer
#model filename
filename = 'reviews_classifier.sav'
loaded_classifier = pickle.load(open(filename, 'rb'))
cv = CountVectorizer(max_features = 500)
#loading the corpus
with open('corpus.data', 'rb') as filehandle:
    # read the data as binary data stream
    corpus = pickle.load(filehandle)
#fitting the count vectorizer with the corpus
cv.fit_transform(corpus)

print("\n---------------------------------------------------")
user_input=input("Enter the review of the restaurant:  ")
test = [user_input]
test_vec = cv.transform(test)
val=loaded_classifier.predict(test_vec)[0]
print("---------------------------------------------------")
if(val==0):
    print("The review entered was negative.")    
    print("The user did not like the restaurant.")
if(val==1):
    print("The review entered was positive.")
    print("The user liked the restaurant.")