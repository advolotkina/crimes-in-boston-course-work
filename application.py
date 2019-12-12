from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import k_means as k_means_module
import hierarchy as hierarchy_module
import simple_tasks as simple_tasks_module
import crimes_freq as crimes_freq_module
import pdfkit

# from flask_weasyprint import HTML, render_pdf

UPLOAD_FOLDER = '/home/zhblnd/crimes_in_boston/flask-server-app/uploads'

# UPLOAD_FOLDER = 'D:/Repositories/IIS/crimes-in-boston-course-work/uploads'
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


@app.route('/k-means')
def k_means():
    k_means_module.optimal_num_of_clusters()
    k_means_module.district_clustering()
    k_means_module.district_and_offence_code_clustering()
    return render_template("k-means.html")


@app.route('/hierarchy')
def hierarchy():
    hierarchy_module.start_hierarhy()
    return render_template("hierarchy.html")


@app.route('/simple-tasks')
def simple_tasks():
    simple_tasks_module.max_shooting_month()
    simple_tasks_module.average_shooting_month()
    return render_template("simple_tasks.html")


@app.route('/сrimes-frequency')
def crimes_frequency():
    crimes_freq_module.crimes_per_year()
    crimes_freq_module.crimes_per_month()
    crimes_freq_module.crimes_per_day_of_week()
    return render_template("crimes_freq.html")


@app.route('/text-review')
def text_review():
    return render_template("text_review.html")


@app.route('/pdf_review')
def pdf_review():
    return render_template("nice_pdf_review.html")


# @app.route('/text_review.pdf')
# def hello_pdf():
#     # Make a PDF from another view
#     return render_pdf(url_for('pdf_review'))
# main loop to run app in debug mode
if __name__ == '__main__':
    app.run(host='0.0.0.0')
