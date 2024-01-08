import numpy as np
import joblib
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load the model
model = joblib.load('nimbus.pkl')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Load the responses
data = joblib.load('responses.pkl')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess patterns
def preprocess_patterns(patterns):
    preprocessed = []
    for pattern in patterns:
        # Tokenize and lemmatize
        words = word_tokenize(pattern)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        # Find synonyms
        synonyms = []
        for word in words:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
        preprocessed.append(' '.join(synonyms))
    return preprocessed

# Predict intent
def predict_intent(text):
    text = preprocess_patterns([text])
    text = vectorizer.transform(text)
    return model.predict(text)[0]

# Get a response
def get_response(intent):
    return np.random.choice(data["intents"][intent]["responses"])

# Get input from the user and respond until the user types "quit"
text = ""
while text != "quit":
    text = input("Enter your message: ")
    intent = predict_intent(text)
    response = get_response(intent)
    print(response)