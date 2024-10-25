# preprocessing.py
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a preprocessing function with lemmatization
def lemmatize_text(text):
    tokens = text.split()  # Simple tokenization by spaces
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)
