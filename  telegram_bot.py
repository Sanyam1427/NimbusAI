import telebot
import logging
import joblib
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

logger = logging.getLogger(__name__)

# Load the models
model = joblib.load('nimbus.pkl')
vectorizer = joblib.load('vectorizer.pkl')
responses = joblib.load('responses.pkl')

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
    return np.random.choice(responses["intents"][intent]["responses"])

# Initialize bot
bot = telebot.TeleBot("6958490839:AAGzx8s-XAbenRNTbccI7YTvDKmDWn73xRw")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, f'Hi {message.from_user.first_name}, welcome! I am Nimbus, an AI powered chatbot which will learn from you :)')

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    text = message.text
    # Predict the intent
    intent = predict_intent(text)
    # Get a response
    response = get_response(intent)
    bot.reply_to(message, response)

bot.polling()