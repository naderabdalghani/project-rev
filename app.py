import logging
import os
import wave

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from pydub import AudioSegment

from app_config import DEBUG, BOT_NAME, TEXT_CHAT_MODE, CACHE_DIR
from keys import FLASK_SECRET_KEY
from core.core import get_bot_response_as_text
from language_model.language_model import load_language_model, correct_user_utterance
from speech_recognizer.speech_recognizer import load_speech_recognizer, wav_to_text
from speech_recognizer.config import SAMPLING_RATE
from core.core import load_core_model

logger = logging.getLogger(__name__)
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
    audio_data = request.files['audio_data']
    audio_data_path = os.path.join(CACHE_DIR, "audio_data")
    audio_data.save(audio_data_path)
    sound = AudioSegment.from_file(audio_data_path)
    sound = sound.set_frame_rate(SAMPLING_RATE)
    wav_file_path = os.path.join(CACHE_DIR, "input.wav")
    sound.export(wav_file_path, format="wav")
    user_utterance = wav_to_text(wav_file_path)
    logger.info(user_utterance)
    corrected_user_utterance = correct_user_utterance(user_utterance)
    logger.info(corrected_user_utterance)
    bot_response = get_bot_response_as_text(corrected_user_utterance)
    logger.info(bot_response)
    return jsonify(bot_response)


@socket_io.on('chat_send')
def handle_user_msg(payload):
    user_utterance = payload['message']
    bot_response = get_bot_response_as_text(user_utterance)
    response = {'name': BOT_NAME, 'reply': bot_response}
    socket_io.emit('chat_response', response)


if __name__ == '__main__':
    socket_io.run(app, debug=DEBUG)
