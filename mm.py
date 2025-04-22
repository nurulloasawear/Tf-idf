from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import math
import io

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Fayl yuklanmadi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Fayl tanlanmadi'}), 400
 
    text = file.read().decode('utf-8', errors='ignore')
    
   
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))  
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    word_counts = Counter(tokens)
    total_words = len(tokens)
    tf_scores = {word: count / total_words for word, count in word_counts.items()}
    
    
    idf_scores = {word: math.log(1 + 1 / (1 + 1)) for word in word_counts}  
    

    tfidf_scores = {word: tf_scores[word] * idf_scores[word] for word in word_counts}
    
  
    sorted_words = sorted(tfidf_scores.items(), key=lambda x: idf_scores[x[0]], reverse=True)[:50]

    result = [
        {'word': word, 'tf': round(tf_scores[word], 4), 'idf': round(idf_scores[word], 4)}
        for word, _ in sorted_words
    ]
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)