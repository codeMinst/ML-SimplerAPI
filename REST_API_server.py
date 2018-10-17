# coding=utf8
##########################################################################################

import os
from flask import Flask, jsonify, json, url_for, send_from_directory
from flask import request
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS, cross_origin
import logging
import logging.handlers
import traceback
import re, datetime
from werkzeug import secure_filename
import UsolDeepCore
from usol import usolUtil

##########################################################################################
app = Flask(__name__)
app.config.from_pyfile('flask.cfg')
# CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

api = Api(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')

    # return "OK"
    return response


def getLogger(logname, logdir, logsize=500 * 1024, logbackup_count=4):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfile = '%s/%s.log' % (logdir, logname)
    loglevel = logging.INFO
    logger = logging.getLogger(logname)
    logger.setLevel(loglevel)
    if logger.handlers is not None and len(logger.handlers) >= 0:
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.handlers = []
    loghandler = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=logsize, backupCount=logbackup_count)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s',
                                  datefmt="%Y-%m-%d %H:%M:%S")
    loghandler.setFormatter(formatter)
    logger.addHandler(loghandler)
    return logger


logger = getLogger('restapi', './log/restapi')

"""
*테스트 검증 요청
@parameter
	1. mode 0: 출근 1: 입실
	2. img  이미지 string
	{mode:"1",
	 img: "DataURL String"
	}
@return
	{"result": "1", "people": [{"name": "LeeSooYon", "prob": "0.999975"}]}
"""


@app.route('/api/validate_face_recognition', methods=['POST'])
def val_fr():
    logger.info("validate_face_recognition.get start...")
    logger.info(request.headers['Content-Type'])
    logger.info(request.json)

    return UsolDeepCore.facePridict(request.json['mode'], request.json['dataURL'])


"""
* Training 요청 
@parameter
	none
@return
	{"eval": "0.999","message": "","result": "1"}
"""


@app.route('/api/request_training', methods=['GET'])
def req_training():
    logger.info("request_training Start")
    logger.info(request.args.get('name'))
    return UsolDeepCore.faceTraining(request.args.get('name'))


"""
*얼굴 등록 요청 
@parameter
	1. name 등록하고자 하는 class
	2. file[]:이미지 배열

@return
	{"result": "1"}
"""


@app.route('/api/register_face', methods=['POST'])
def register_face():
    logger.info("register_face  start...")

    logger.info(request.form['name'])

    create_new_folder(app.config['UPLOAD_FOLDER'] + "/" + request.form['name'])
    # Get the name of the uploaded files
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            # Move the file form the temporal folder to the upload
            # folder we setup
            file.save(os.path.join(app.config['UPLOAD_FOLDER'] + "/" + request.form['name'], filename))
            logger.info(request.form['name'])
            # Save the filename into a list, we'll use it later
            filenames.append(filename)
    # Redirect the user to the uploaded_file route, which
    # will basicaly show on the browser the uploaded file
    # Load an html page with a link to each uploaded file
    send_from_directory(app.config['UPLOAD_FOLDER'] + "/" + request.form['name'], filename, as_attachment=True)

    result = {'result': '1', 'msg': "OK"}
    jsonString = json.dumps(result)
    return jsonString


"""
*얼굴 등록 폴더 생성  요청 
@parameter
	1. name 등록하고자 하는 class
	{name:"park"
	}
@return
	{'result': '1', 'msg': "OK"}
"""


@app.route('/api/register_faces', methods=['POST'])
def register_faces():
    logger.info("register_faces  start...")
    logger.info(request.headers['Content_Type'])
    logger.info(request.json)

    return UsolDeepCore.faceRegister(request.json['name'])


"""
*얼굴 이미지 등록 요청 
@parameter
	1. name 등록하고자 하는 class
	2. dataURL: "DataURL String"
	{name:"park",
	 dataURL: "DataURL String"
	}
@return
	result = {'result': '1', 'msg': "OK"}
"""


@app.route('/api/register_images', methods=['POST'])
def register_images():
    logger.info("register_images  start...")
    logger.info(request.json)

    if not request.json['name'] or not request.json['dataURL']:
        result = {'result': '0', 'msg': "Parameter error"}
    else:

        strImg = request.json['dataURL']
        recImg = usolUtil.bas64ToRGB(strImg)

        basename = "image"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        filetype = ".jpg"
        img_name = "_".join([basename, suffix]) + filetype

        saved_path = os.path.join(app.config['UPLOAD_FOLDER'] + request.json['name'] + "/", img_name)
        app.logger.info("saving {}".format(saved_path))
        recImg.save(saved_path)

        result = {}
        if os.path.isfile(saved_path):

            result = {'result': '1', 'msg': "OK"}
        else:
            result = {'result': '0', 'msg': "image upload error"}

    jsonString = json.dumps(result)

    return jsonString


class CreateUser(Resource):
    def post(self):
        try:
            logger.info("CreateUser start...")
            raise Exception('spam', 'eggs')
            return {'message': 'try'}
        except Exception:
            return {'message': 'except'}
        finally:
            logger.info("CreateUser end...")
            return {'message': 'finally'}


api.add_resource(CreateUser, '/user')

if __name__ == '__main__':
    app.run(host='172.17.2.35', debug=True)
