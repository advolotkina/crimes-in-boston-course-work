from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/home/zhblnd/crimes_in_boston/flask-server-app/uploads'
ALLOWED_EXTENSIONS = set(['csv'])
# Create a new Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('data'))
    return render_template("main.html")

@app.route('/data')
def data():
    return render_template("data.html")

@app.route('/graph-reviews')
def graph_reviews():
    return render_template("graph_reviews.html")

@app.route('/text-reviews')
def text_reviews():
    return render_template("text_reviews.html")

# main loop to run app in debug mode
if __name__ == '__main__':
    app.run(host='0.0.0.0')