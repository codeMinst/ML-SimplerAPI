# coding=utf8
"""
안면인식을 위한 restAPI를 flask로 구현하였고,
간단한 API사용 웹클라이언트 샘플페이지를 트레이닝 이미지 전송 편의상 같은 flask서버,템플릿으로 구성하였다.
다른 클라이언트 환경에서 트레이닝 이미지 전송을 받기 위한 클래스 디렉토리 생성 및 전송 API도 마련해 두었다.
"""
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
app.config['UPLOAD_FOLDER'] = 'dataset'
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
*이미지 등록 디렉토리 생성요청 
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
*이미지 전송요청 
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
*Classification test
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

# https://<<domain>>:<<port>>/api/request_training
# Params:
#   name - 모델에 추가할 클래스(사람) 이름
# Return values:
#   1 - 디렉토리 생성 성공
#   0 - 디렉토리 생성 실패
# Actions:
#   name 디렉토리 생성
def createDir(name):
    result = {}
    if not os.path.exists(app.config['UPLOAD_FOLDER'] + name):
        os.makedirs(app.config['UPLOAD_FOLDER'] + name)
        result = {'result': '1', 'msg': "OK"}
    else:
        result = {'result': '0', 'msg': "Name(" + name + ") already exists"}

    jsonString = json.dumps(result)
    return jsonString

@app.route('/upload', methods=['POST'])
def upload():
    """샘플페이지 flask form 이미지 업로드 function.

    # Arguments
        className:분류될 라벨 값
        file[]:복수의 이미지 데이터

    # Returns
       front 웹페이지 템플릿
    """
    className = request.form.get('className')
    uploadFiles = request.files.getlist("file[]")

    dirPath = os.path.join(app.config['UPLOAD_FOLDER'], className)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
        for file in uploadFiles:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(dirPath, filename))
        result = render_template('upload.html', result=getTraningDirFiles())
    else:
        result = "Name(" + className + ") already exists"

    return result

@app.route('/upload')
def showUploadPage():
    return render_template('upload.html', result=getTraningDirFiles())

@app.route('/uploaded/<className>/<fileName>')
def uploadedFile(className, fileName):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], className), fileName)

@app.route('/')
def serverOn():
    return render_template("index.html")

def getTraningDirFiles():
    classNames = util.getFilesDir(app.config['UPLOAD_FOLDER'])
    files = []
    for className in classNames:
        for file in util.getFilesDir(os.path.join(app.config['UPLOAD_FOLDER'], className)):
            files.append([className, file])
    return  files

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
