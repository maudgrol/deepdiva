from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
import time as time 

# Dummy data
BOOKS = [
    {
        'title': 'On the Road',
        'author': 'Jack Kerouac',
        'read': True
    },
    {
        'title': 'Harry Potter and the Philosopher\'s Stone',
        'author': 'J. K. Rowling',
        'read': False
    },
    {
        'title': 'Green Eggs and Ham',
        'author': 'Dr. Seuss',
        'read': True
    }
]

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# add h2p file to app.config
app.config["data_folder"] = "/Users/grahamherdman/Documents/data-science-retreat/deep-diva/deepdiva/data/"

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

# prediction route
@app.route('/prediction', methods=['POST'])
def prediction():

    post_data = request.get_json()
    # import pdb; pdb.set_trace()

    time.sleep(5)

    h2p_filename = "HS-Brighton.h2p"

    try:
        return send_from_directory(app.config["data_folder"], filename=h2p_filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)

# books route
@app.route('/books', methods=['GET', 'POST'])
def all_books():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        post_data = request.get_json()
        BOOKS.append({
            'title': post_data.get('title'),
            'author': post_data.get('author'),
            'read': post_data.get('read')
        })
        response_object['message'] = 'Book added!'
    else:
        response_object['books'] = BOOKS
    return jsonify(response_object)

# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

if __name__ == '__main__':
    app.run()
