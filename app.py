from flask import Flask, render_template
from flask_socketio import SocketIO
from utilities.config import DEBUG
from core.core import get_bot_response_as_text

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socket_io = SocketIO(app)


@app.route('/')
def sessions():
    return render_template('text_chat_session.html')


@socket_io.on('chat_send')
def handle_user_msg(payload):
    user_utterance = payload['message']
    bot_response = get_bot_response_as_text(user_utterance)
    response = {'name': 'Ted', 'reply': bot_response}
    socket_io.emit('chat_response', response)


if __name__ == '__main__':
    socket_io.run(app, debug=DEBUG)
