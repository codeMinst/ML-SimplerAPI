# coding=utf8
import datetime
import logging
import logging.handlers
import os
from flask import Flask, json, send_from_directory, render_template
from flask import request
from flask_cors import CORS
from flask_restful import Api, Resource
from utils import util
from werkzeug.utils import secure_filename
from api_face import UsolDeepCore

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


app = Flask(__name__)
# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'wav', 'webm'])
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api = Api(app)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    # return "OK"
    return response

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
        recImg = util.bas64ToRGB(strImg)

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

'''
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
'''

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded files
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return render_template('upload.html', filenames=filenames)

@app.route('/upload')
def showUploadPage():
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def serverOn():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
