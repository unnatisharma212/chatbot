from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# In-memory chat history (list of dicts)
chat_history = []

def get_timestamp():
    return datetime.now().strftime('%H:%M')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    if not user_msg:
        return jsonify({'error': 'No message provided'}), 400
    # Save user message
    chat_history.append({'sender': 'user', 'text': user_msg, 'timestamp': get_timestamp()})
    # Generate bot reply (replace with your real bot logic)
    bot_reply = f"You said: {user_msg}"
    chat_history.append({'sender': 'bot', 'text': bot_reply, 'timestamp': get_timestamp()})
    return jsonify({'reply': bot_reply})

@app.route('/history', methods=['GET'])
def history():
    return jsonify({'history': chat_history})

if __name__ == '__main__':
    app.run(debug=True) 