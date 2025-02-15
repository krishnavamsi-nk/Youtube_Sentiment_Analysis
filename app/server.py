from flask import Flask, request, render_template
from .predict import predict_sentiments
from .youtube import get_video_comments
from flask_cors import CORS
import os
import re

app = Flask(__name__)
CORS(app)

def extract_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    print(f"🔍 Extracting Video ID from: {url}")  # Debugging
    
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",  # Matches standard YouTube video IDs
        r"(?:youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})"  # Matches shortened or embedded links
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            print(f"✅ Extracted Video ID: {video_id}")  # Debugging
            return video_id
    
    print("⚠️ Invalid YouTube URL!")  # Debugging
    return None

def get_video(video_id):
    print(f"🔍 Fetching comments for video: {video_id}")  # Debugging

    if not video_id:
        return {"error": "Invalid Video ID"}

    comments = get_video_comments(video_id)
    print("Fetched Comments:", comments)  # Debugging
    
    if not comments:
        return {"comments": [], "message": "No Comments Present"}

    predictions = predict_sentiments(comments)
    positive = predictions.count("Positive")
    negative = predictions.count("Negative")

    summary = {
        "positive": positive,
        "negative": negative,
        "num_comments": len(comments),
        "rating": (positive / len(comments)) * 100 if len(comments) > 0 else 0
    }

    print(f"✅ Analysis Complete: {summary}")  # Debugging
    return {"predictions": predictions, "comments": comments, "summary": summary}

@app.route('/', methods=['GET', 'POST'])
def index():
    print("📌 Request received at '/' route")  # Debugging

    summary = None
    comments = []
    
    if request.method == 'POST':
        video_url = request.form.get('video_url', "").strip()
        print(f"📺 Video URL Received: {video_url}")  # Debugging
        
        video_id = extract_video_id(video_url)
        if video_id:
            data = get_video(video_id)
            if "error" not in data:
                summary = data['summary']
                comments = list(zip(data['comments'], data['predictions']))
            else:
                print("⚠️ Error Fetching Data:", data["error"])  # Debugging

    return render_template('index.html', summary=summary, comments=comments)

if __name__ == '__main__':
    print("🚀 Flask server is starting...")  # Debugging
    port = int(os.environ.get("PORT", 10000))  # Use Render's dynamic PORT
    app.run(host="0.0.0.0", port=port, debug=True)
