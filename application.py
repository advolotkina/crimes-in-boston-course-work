from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import k_means as k_means_module
import hierarchy as hierarchy_module
import simple_tasks as simple_tasks_module
import crimes_freq as crimes_freq_module
import clean_data as clean_data_module
from PIL import Image
import PIL
import pdfkit
from flask_weasyprint import HTML, render_pdf
from sklearn import metrics

UPLOAD_FOLDER = '/home/zhblnd/crimes_in_boston/flask-server-app/uploads'
ALLOWED_EXTENSIONS = set(['csv','pkl'])
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
            return redirect(url_for('data', filename = filename ))
    return render_template("main.html")

@app.route('/data')
def data():
    filename = request.args.get('filename')
    return render_template("data.html", filename = filename)

@app.route('/user-task')
def user_task():
    return render_template("user_task.html")

@app.route('/user-task/predict_with_model', methods=['GET','POST'])
def predict_with_model():
    if request.method == 'POST':
        file = request.files['file']
        model_file = request.files['model']
        model_file2 = request.files['model2']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if model_file and allowed_file(model_file.filename):
                model_filename = secure_filename(model_file.filename)
                model_file.save(os.path.join(app.config['UPLOAD_FOLDER'], model_filename))
                if model_file2 and allowed_file(model_file2.filename):
                    model_filename2 = secure_filename(model_file2.filename)
                    model_file2.save(os.path.join(app.config['UPLOAD_FOLDER'], model_filename2))
                    option = request.form['radiobutton']
                    print(option)
                    return redirect(url_for('predict_with_model_main', data=filename, model=model_filename, model2 = model_filename2,
                                            clustering = option))
    return render_template("predict_with_model.html")

@app.route('/user-task/predict_with_model/main')
def predict_with_model_main():
    data_file = request.args.get('data')
    model_file = request.args.get('model')
    model_file2 = request.args.get('model2')
    clean_data_module.clean(data_file)
    clustering = request.args.get('clustering')
    if clustering == "k_means":
        clean_data_module.clean(data_file)
        score, score2, objects, labels = k_means_module.predict_with_model(data_file, model_file, model_file2)
        dictionary = dict(zip(tuple(objects), tuple(labels)))
        return render_template("k-means.html", score=score, score2=score2, _dict = dictionary, model_filename = model_file,
                               model_filename2 = model_file2, filename = data_file)
    if clustering == "hierarchy":
        return "TODO"


@app.route('/user-task/train_and_predict', methods=['GET','POST'])
def train_and_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('train_and_predict_main', filename = filename ))
    return render_template("train_and_predict.html")

@app.route('/user-task/train_and_predict/main')
def train_and_predict_main():
    data_file = request.args.get('filename')
    clean_data_module.clean(data_file)
    k_means_score1, k_means_score2, objects_array, labels_array, model_filename, model_filename2 = k_means_module.start_k_means(data_file)
    hierarchy_score1, hierarchy_score2 = hierarchy_module.start_hierarhy(data_file)
    if (1-k_means_score1)<(1-hierarchy_score1):
        dictionary = dict(zip(tuple(objects_array), tuple(labels_array)))

        return render_template("best_k_means.html", k_means_score = k_means_score1, hierarchy_score = hierarchy_score1,
                               score = k_means_score1, score2 = k_means_score2, _dict = dictionary,
                               filename = data_file, model_filename = model_filename, model_filename2 = model_filename2)
    else:
        return render_template("best_hierarchy.html")


@app.route('/user-task/train_and_predict/k_means<a>')
@app.route('/user-task/train_and_predict/k_means<a>/<b>')
@app.route('/user-task/train_and_predict/k_means<a>/<b>/<c>')
@app.route('/user-task/train_and_predict/k_means')
def best_k_means_pdf():
    score = request.args.get('_score')
    score2 = request.args.get('_score2')
    filename = request.args.get('file')
    hierarchy_score = request.args.get('hierarchy_score')
    return render_pdf(url_for('nice_k_means_pdf', score = score, score2 = score2, file = filename,
                              hierarchy_score = hierarchy_score))

@app.route('/best_k_means_pdf_review')
def nice_k_means_pdf():
    score = request.args.get('score')
    score2 = request.args.get('score2')
    filename = request.args.get('file')
    hierarchy_score = request.args.get('hierarchy_score')
    dictionary = k_means_module.get_elements_and_labels(filename)
    im = Image.open("./static/district_and_offence_code_clustering_k_means.png")
    width, height = im.size
    newsize = (700, 700)
    im = im.resize(newsize)
    im.save("./static/district_and_offence_code_clustering_k_means_review.png")

    return render_template("nice_best_k_means_review.html", score = score, score2 = score2, _dict = dictionary,
                           hierarchy_score = hierarchy_score, k_means_score = score)

@app.route('/download-model')
def download_model():
    model_filename = request.args.get('model_filename')
    return send_from_directory('./models/',
                               model_filename, as_attachment=True)

@app.route('/k-means',methods=['GET','POST'])
def k_means():
    filename = request.args.get('filename')
    clean_data_module.clean(filename)
    score, score2, objects, labels, model_filename, model_filename2 = k_means_module.start_k_means(filename)
    dictionary = dict(zip(tuple(objects), tuple(labels)))

    return render_template("k-means.html", score = score, score2 = score2, _dict = dictionary,
                           model_filename = model_filename, model_filename2 = model_filename2,
                           filename = filename)


@app.route('/hierarchy',methods=['GET','POST'])
def hierarchy():
    filename = request.args.get('filename')
    clean_data_module.clean(filename)
    hierarchy_module.start_hierarhy(filename)
    return render_template("hierarchy.html")


@app.route('/simple-tasks',methods=['GET','POST'])
def simple_tasks():
    filename = request.args.get('filename')
    month_with_max_shootings = simple_tasks_module.max_shooting_month(filename)
    average_shooting_month = simple_tasks_module.average_shooting_month(filename)
    return render_template("simple_tasks.html", average_shooting_month = average_shooting_month,
                           month_with_max_shootings = month_with_max_shootings, filename = filename)

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
    f = open("./scores/"+"k_means_score", "r+")
    score = f.read()
    f.close()
    print(score)
    f = open("./scores/"+"k_means_score_2", "r+")
    score_2 = f.read()
    f.close()
    print(score_2)
    im = Image.open("./static/hierarchy_clustering.png")
    width, height = im.size
    newsize = (700, 700)
    im = im.resize(newsize)
    im.save("./static/hierarchy_clustering_review.png")
    f = open("./scores/"+"hierarchy_score_1", "r+")
    hierarchy_score = f.read()
    f.close()

    return render_template("nice_pdf_review.html",k_means_score = score, k_means_score2 = score_2,
                           hierarchy_score = hierarchy_score )

@app.route('/simple_review/')
def simple_review():
    filename = request.args.get('file')
    month_with_max_shootings = simple_tasks_module.max_shooting_month(filename)
    average_shooting_month = simple_tasks_module.average_shooting_month(filename)
    shooting_list = simple_tasks_module.getShootingNumPerMonth(filename)
    i = 1;
    return render_template("simple_review.html",average_shooting_month = average_shooting_month,
                           month_with_max_shootings = month_with_max_shootings, shooting_list = shooting_list,i = i)

# Не текстовый, а общий(
@app.route('/text_review.pdf')
def hello_pdf():
    # Make a PDF from another view
    return render_pdf(url_for('pdf_review'))

@app.route('/simple_review.pdf',)
def simple_pdf():
    filename = request.args.get('file')
    return render_pdf(url_for("simple_review",file = filename))
# main loop to run app in debug mode
if __name__ == '__main__':
    app.run(host='0.0.0.0')