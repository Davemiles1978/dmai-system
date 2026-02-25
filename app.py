from flask import Flask, send_from_directory
import os

app = Flask(__name__, static_folder='ui')

@app.route('/')
def serve_ui():
    return send_from_directory('ui', 'ai_ui.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('ui', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
