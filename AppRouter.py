from flask import Flask, request, jsonify, send_file
from EdgeTTSModel import synthesize_speech
import os

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to LinkAI.\nText to Speech Part."


@app.route('/tts', methods=['POST'])
async def convert_text_to_voice():
    text = request.headers['text']
    audio_path, audio_file_name = await synthesize_speech(text=text)

    if os.path.exists(audio_path):
        current_directory = os.path.dirname(os.path.abspath(audio_path))
        return send_file(f"{current_directory}\{audio_file_name}", as_attachment=True, download_name="generated_audio.mp3")
    else:
        return jsonify({'error': 'Failed to generate audio'}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
