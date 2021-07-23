import os

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from app_config import DEBUG, BOT_NAME, TEXT_CHAT_MODE, CACHE_DIR
from keys import FLASK_SECRET_KEY
from core.core import get_bot_response_as_text
from language_model.language_model import load_language_model
from speech_recognizer.speech_recognizer import load_speech_recognizer
from core.core import load_core_model

app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
socket_io = SocketIO(app)


@app.route('/')
def sessions():
    if TEXT_CHAT_MODE:
        return render_template('text_chat_session.html')
    return render_template('voice_chat_session.html')


@app.route('/initialize')
def initialize():
    load_speech_recognizer()
    load_core_model()
    load_language_model()
    return jsonify("Models loaded successfully")


@app.route('/send_wav', methods=['POST'])
def handle_user_wav():
    wav_file = request.files['audio_data']
    save_path = os.path.join(CACHE_DIR, "input.wav")
    wav_file.save(save_path)
    response = "test"
    return jsonify(response)


@socket_io.on('chat_send')
def handle_user_msg(payload):
    user_utterance = payload['message']
    bot_response = get_bot_response_as_text(user_utterance)
    response = {'name': BOT_NAME, 'reply': bot_response}
    socket_io.emit('chat_response', response)


if __name__ == '__main__':
    socket_io.run(app, debug=DEBUG)
