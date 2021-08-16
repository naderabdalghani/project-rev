import logging
import os
import warnings

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from pydub import AudioSegment
from silence_tensorflow import silence_tensorflow

from app_config import DEBUG, BOT_NAME, CACHE_DIR, INPUT_FILENAME, APP_MODE, APP_MODES
from core.core import get_bot_response_as_text, load_core_model
from keys import FLASK_SECRET_KEY
from language_model.language_model import correct_user_utterance, load_language_model
from speech_recognizer.config import SAMPLING_RATE
from speech_recognizer.speech_recognizer import wav_to_text, load_speech_recognizer
from speech_synthesizer.speech_synthesizer import render_bot_response_as_audio, load_speech_synthesizer

silence_tensorflow()
warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
socket_io = SocketIO(app)


@app.route('/')
def sessions():
    if APP_MODE == APP_MODES["VOICE_CHAT_LITE_MODE"]:
        return render_template('voice_chat_lite_session.html')
    if APP_MODE == APP_MODES["VOICE_CHAT_MODE"]:
        return render_template('voice_chat_session.html')
    return render_template('text_chat_session.html')


@app.route('/initialize')
def initialize():
    if APP_MODE == APP_MODES["VOICE_CHAT_MODE"]:
        load_speech_recognizer()
        load_language_model()
        load_speech_synthesizer()
    if APP_MODE == APP_MODES["VOICE_CHAT_LITE_MODE"]:
        load_speech_synthesizer()
    load_core_model()
    return jsonify("Models loaded successfully")


@app.route('/send_wav', methods=['POST'])
def handle_user_wav():
    audio_data = request.files['audio_data']
    audio_data_path = os.path.join(CACHE_DIR, "audio_data")
    audio_data.save(audio_data_path)
    sound = AudioSegment.from_file(audio_data_path)
    sound = sound.set_frame_rate(SAMPLING_RATE)
    wav_file_path = os.path.join(CACHE_DIR, INPUT_FILENAME)
    sound.export(wav_file_path, format="wav")
    user_utterance = wav_to_text(wav_file_path)
    logger.info(user_utterance)
    corrected_user_utterance = correct_user_utterance(user_utterance)
    logger.info(corrected_user_utterance)
    bot_response = get_bot_response_as_text(corrected_user_utterance)
    logger.info(bot_response)
    render_bot_response_as_audio(bot_response)
    return jsonify(bot_response)


@app.route('/send_text', methods=['POST'])
def handle_user_text():
    user_utterance = request.data.decode("utf-8")
    logger.info(user_utterance)
    bot_response = get_bot_response_as_text(user_utterance)
    logger.info(bot_response)
    render_bot_response_as_audio(bot_response)
    return jsonify(bot_response)


@socket_io.on('chat_send')
def handle_user_msg(payload):
    user_utterance = payload['message']
    bot_response = get_bot_response_as_text(user_utterance)
    response = {'name': BOT_NAME, 'reply': bot_response}
    socket_io.emit('chat_response', response)


if __name__ == '__main__':
    socket_io.run(app, debug=DEBUG)
