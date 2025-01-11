from flask import Flask, request, render_template, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset
try:
    df = pd.read_csv('train.tsv', sep='\t', encoding='utf-8')
    
    # Define the columns we need
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state', 'party', 
              'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 
              'pants_fire_counts', 'context']
    df.columns = columns
    
    # Map the different truth values to simplified categories
    label_mapping = {
        'true': 'TRUE',
        'mostly-true': 'TRUE',
        'half-true': 'MIXED',
        'barely-true': 'FALSE',
        'false': 'FALSE',
        'pants-fire': 'FALSE'
    }
    df['label'] = df['label'].map(label_mapping)
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame({
        'statement': ['This is a sample statement'],
        'label': ['TRUE']
    })

def preprocess_text(text):
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    words = [word for word in text.split() if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# Use 'statement' column instead of 'text'
df['text_cleaned'] = df['statement'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text_cleaned'], df['label'], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        processed_text = preprocess_text(news_text)
        text_tfidf = vectorizer.transform([processed_text])
        prediction = model.predict(text_tfidf)
        
        # Map prediction to more user-friendly output
        result_mapping = {
            'TRUE': 'LIKELY TRUE',
            'MIXED': 'MIXED/UNCERTAIN',
            'FALSE': 'LIKELY FALSE'
        }
        result = result_mapping.get(prediction[0], 'UNCERTAIN')
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)