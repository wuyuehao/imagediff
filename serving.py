from flask import Flask, jsonify, request, render_template, redirect, url_for
import pandas as pd
import traceback
from scipy import spatial
from werkzeug.utils import secure_filename
from flask_dropzone import Dropzone
from flask_dropzone.utils import random_filename
from category_match import img_classification, img_similarity
import os
from pixel_diff import simple_pixel_diff
from feature_match import feature_matching, feature_diff


def cosine_smilarity(v1, v2):
    cosine_similarity = 1 - spatial.distance.cosine(v1, v2)
    return cosine_similarity


def create_app(config_filename=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    basedir = os.path.abspath(os.path.dirname(__file__))

    app.config.update(
        UPLOADED_PATH=os.path.join(basedir, 'static'),
        # Flask-Dropzone config:
        DROPZONE_ALLOWED_FILE_TYPE='image',
        DROPZONE_MAX_FILE_SIZE=2,
        DROPZONE_MAX_FILES=2,
        DROPZONE_UPLOAD_ON_CLICK=True,
    )

    dropzone = Dropzone(app)
    result = {}
    app.config["DEBUG"] = True
    if config_filename:
        app.config.from_pyfile(config_filename)

    def similarity(f1_name, f2_name):
        try:

            f1 = os.path.join(app.config['UPLOADED_PATH'], f1_name)
            f2 = os.path.join(app.config['UPLOADED_PATH'], f2_name)
            id = f1_name[:8] + f2_name[:8] + ".png"
            pixel_match_file = os.path.join(app.config['UPLOADED_PATH'], "PM_" + id)
            feature_match_file = os.path.join(app.config['UPLOADED_PATH'], "FM_" + id)
            feature_diff_file1 = os.path.join(app.config['UPLOADED_PATH'], "FD1_" + id)
            feature_diff_file2 = os.path.join(app.config['UPLOADED_PATH'], "FD2_" + id)
            catagory_match_file = os.path.join(app.config['UPLOADED_PATH'], "CM_" + id)
            app.logger.debug('comparing %s with %s', f1, f2)
            pd = simple_pixel_diff(f1, f2, pixel_match_file)
            app.logger.debug('pixel matching score: %f', pd)
            feature_match_score = feature_matching(f1, f2, feature_match_file)
            app.logger.debug('feature matching score: %d', feature_match_score)

            feature_diff_score = feature_diff(f1, f2, feature_diff_file1, feature_diff_file2)

            category1 = img_classification(f1)
            category2 = img_classification(f2)

            euclidean_distance, cosine_distance= img_similarity(f1,f2)


            return {'f1': f1_name, 'f2': f2_name, 'success': True, 'pixel_match_score': pd,
                    'pixel_match_img': "PM_" + id, 'feature_match_score': feature_match_score,
                    'feature_match_img': "FM_" + id, 'euclidean_distance': euclidean_distance, 'cosine_distance': cosine_distance,
                    'category_match_img': "CM_" + id, 'feature_diff_score': feature_diff_score,
                    'feature_diff_img1' : "FD1_" + id, 'feature_diff_img2' : "FD2_" + id, "category1":category1, "category2":category2}
        except:
            traceback.print_exc()
            return {'success': False, 'utterances': None}

    @app.route('/', methods=['GET'])
    def home():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload():
        files = []
        if request.method == 'POST':
            for key, f in request.files.items():
                if key.startswith('file'):
                    name = random_filename(f.filename)
                    f.save(os.path.join(app.config['UPLOADED_PATH'], name))
                    files.append(name)
                    app.logger.info('saving %s', f.filename)
        app.logger.debug('received %d', len(files))

        return jsonify({"files": files})

    @app.route('/diff', methods=['GET'])
    def completed():
        f1 = request.args.get('f1')
        f2 = request.args.get('f2')
        res = similarity(f1, f2)

        return render_template('response.html', res=res)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
