import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer



stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def s_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return s_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


print (cosine_sim('hey fatima zehra here', 'i am python developer'))
print (cosine_sim('i am a student', 'i am doing python internship'))
print (cosine_sim('by CodexCue ', ' by codexcue'))