from flask import Flask, request, render_template
from predict import predict_sentiments
from youtube import get_video_comments
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_video(video_id):
    print(f"🔍 Fetching comments for video: {video_id}")  # Debugging

    if not video_id:
        return {"error": "video_id is required"}

    comments = get_video_comments(video_id)
    print("Fetched Comments:", comments)  # Debugging
    if not comments:  # If no comments found
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
        video_url = request.form.get('video_url')
        print(f"📺 Video URL Received: {video_url}")  # Debugging
        
        if "v=" in video_url:
            video_id = video_url.split("v=")[1]
            data = get_video(video_id)

            summary = data['summary']
            comments = list(zip(data['comments'], data['predictions']))
        else:
            print("⚠️ Invalid Video URL Format!")  # Debugging

    return render_template('index.html', summary=summary, comments=comments)

if __name__ == '__main__':
    print("🚀 Flask server is starting...")  # Debugging
    app.run(debug=True)
