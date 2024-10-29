from flask import Flask, request, jsonify, send_file
from EdgeTTSModel import synthesize_speech
import os
import asyncio

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to LinkAI.\nText to Speech Part."


@app.route('/tts', methods=['POST'])
async def convert_text_to_voice():
    text = request.headers['text']
    audio_path = await synthesize_speech(text=text)

    if os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=True, download_name="generated_audio.mp3")
    else:
        return jsonify({'error': 'Failed to generate audio'}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)
