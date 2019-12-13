from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import k_means as k_means_module
import hierarchy as hierarchy_module
import simple_tasks as simple_tasks_module
import crimes_freq as crimes_freq_module
import pdfkit
from flask_weasyprint import HTML, render_pdf
from sklearn import metrics

UPLOAD_FOLDER = '/home/zhblnd/crimes_in_boston/flask-server-app/uploads'
ALLOWED_EXTENSIONS = set(['csv'])
# Create a new Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

default_filename = "tmpex0j7dw9.csv"

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            default_filename = filename
            return redirect(url_for('data', filename = filename ))
    return render_template("main.html")

@app.route('/data')
def data():
    filename = request.args.get('filename')
    print(filename)
    return render_template("data.html", filename = filename)

@app.route('/k-means',methods=['GET','POST'])
def k_means():
    filename = request.args.get('filename')
    print(filename)
    k_means_module.start_k_means(filename)
    return render_template("k-means.html")

@app.route('/hierarchy',methods=['GET','POST'])
def hierarchy():
    filename = request.args.get('filename')
    hierarchy_module.start_hierarhy(filename)
    return render_template("hierarchy.html")


@app.route('/simple-tasks',methods=['GET','POST'])
def simple_tasks():
    filename = request.args.get('filename')
    month_with_max_shootings = simple_tasks_module.max_shooting_month(filename)
    average_shooting_month = simple_tasks_module.average_shooting_month(filename)
    default_filename = filename
    return render_template("simple_tasks.html", average_shooting_month = average_shooting_month,
                           month_with_max_shootings = month_with_max_shootings)

@app.route('/сrimes-frequency',methods=['GET','POST'])
def crimes_frequency():
    filename = request.args.get('filename')
    crimes_freq_module.crimes_freq_start(filename)
    return render_template("crimes_freq.html")

@app.route('/text-review')
def text_review():
    return render_template("text_review.html")

@app.route('/pdf_review')
def pdf_review():
    return render_template("nice_pdf_review.html")

@app.route('/simple_review/')
def simple_review():
    # default_filename = request.args.get('filename')
    month_with_max_shootings = simple_tasks_module.max_shooting_month(default_filename)
    average_shooting_month = simple_tasks_module.average_shooting_month(default_filename)
    shooting_list = simple_tasks_module.getShootingNumPerMonth(default_filename)
    i = 1;
    return render_template("simple_review.html",average_shooting_month = average_shooting_month,
                           month_with_max_shootings = month_with_max_shootings, shooting_list = shooting_list,i = i)

@app.route('/text_review.pdf')
def hello_pdf():
    # Make a PDF from another view
    return render_pdf(url_for('pdf_review'))

@app.route('/simple_review.pdf')
def simple_pdf():
    return render_pdf(url_for("simple_review"))
# main loop to run app in debug mode
if __name__ == '__main__':
    app.run(host='0.0.0.0')