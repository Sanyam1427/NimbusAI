import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import joblib

nltk.download('punkt')
nltk.download('wordnet')

# Define intents, patterns, and responses
data = {
    "intents": {
        "greeting": {
            "patterns": ["Hello", "Hi", "Greetings", "Hey there", "What's up", "yo"],
            "responses": ["Hello", "Hi", "Hey there", "Hi boss! What's up?"]
        },
        "bye": {
            "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time", "adios", "adios amigo", "see ya", "ttyl", "talk to you later", "goodnight", "have a nice day", "take care", "see you soon", "farewell", "catch you later", "peace out", "see you later", "smell you later", "have to go", "i'm off", "have to run", "pitter patter", "gotta go", "i'm out of here", "i'm out", "i'm off", "tata"],
            "responses": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time", "adios", "adios amigo", "see ya", "ttyl", "talk to you later", "goodnight", "have a nice day", "take care", "see you soon", "farewell", "catch you later", "peace out", "see you later", "smell you later", "have to go", "have to run", "pitter patter", "gotta go", "i'm out of here", "i'm out", "i'm off", "tata"]
        },
        "gratitude": {
            "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me", "I owe you one", "Cheers mate", "Much appreciated", "helful" , "appreciated"],
            "responses": ["Happy to help!", "Any time!", "My pleasure", "You're most welcome!"]
        },
        "mood": {
            "patterns": ["How are you?", "How are you doing?", "What's going on?", "How is your day?", "How's everything?", "How's life?", "How's your day going?", "How have you been?", "What's up?", "What's new?", "What's happening?", "How's it going?"],
            "responses": ["I'm doing great. Thanks for asking!", "I'm doing good. Thanks for asking!", "I'm doing well. Thanks for asking!", "I'm doing awesome. Thanks for asking!", "I'm doing fine. Thanks for asking!", "I'm doing okay. Thanks for asking!", "I'm doing alright. Thanks for asking!", "I'm doing good. Thanks for asking!", "I'm doing great. Thanks for asking!", "I'm doing well. Thanks for asking!", "I'm doing awesome. Thanks for asking!", "I'm doing fine. Thanks for asking!", "I'm doing okay. Thanks for asking!", "I'm doing alright. Thanks for asking!"]
        },
        "anger": {
            "patterns": ["You suck", "You're a loser", "You're a failure", "You're a disappointment", "You're a waste of time", "You're a waste of space", "You're a waste of oxygen", "You're a waste of life", "You're a waste of a human being", "You're a waste of a person", "You're a waste of a soul", "You're a waste of a body", "You're a waste of a mind", "You're a waste of a brain"],
            "responses": ["I'm sorry you feel that way", "I'm sorry you feel that way. I'll try to do better", "I'm sorry you feel that way. I'll try to do better next time", "I'm sorry you feel that way. I'll try to do better next time", "I'm sorry you feel that way. I'll try to do better next time", "I'm sorry you feel that way. I'll try to do better next time", "I'm sorry you feel that way. I'll try to do better next time", "I'm sorry you feel that way. I'll try to do better next time", "I'm sorry you feel that way. I'll try to do better next time", "I'm sorry you feel that way. I'll try to do better next time"]
        },
        "happy": {
            "patterns": ["I'm happy", "I'm feeling good", "I'm excited", "I'm thrilled", "This is awesome", "I'm satisfied", "I'm pleased", "I love this"],
            "responses": ["That's great to hear!", "I'm glad you're happy!", "That's awesome!", "I'm happy to hear that!"]
        }
    }
}

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

# Create training set
X_train = []
y_train = []
for intent, value in data["intents"].items():
    patterns = preprocess_patterns(value["patterns"])
    X_train.extend(patterns)
    y_train.extend([intent] * len(patterns))

# Vectorize patterns
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# Train SVC
model = SVC()
model.fit(X_train, y_train)

# Predict intent
def predict_intent(text):
    text = preprocess_patterns([text])
    text = vectorizer.transform(text)
    return model.predict(text)[0]

# Save the model
joblib.dump(model, 'nimbus.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Save the responses
joblib.dump(data, 'responses.pkl')