
import json
import urllib

config = None
with open("./config.json",'r') as load_f:
    config = json.load(load_f)
    
from flask import Flask,request,jsonify
from flask import render_template
import jwt
import time
import random
app = Flask(__name__)

from deploy.intention_classify import intention
from deploy.sementic_similarity import corpus
from deploy.clawer import claw_answer

status_code = {
    'success': 0,
    'fail': -1,
    'invalid_token': -2
}

# 跨域支持
def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def create_token(request):
    grant_type = request.json.get('grant_type')
    #如果是新会话，使用随机数创建一个token
    key = str(random.random())
    payload = {
        "iss": "gusibi.com",
         "iat": int(time.time()),
         "exp": int(time.time()) + 86400 * 7,
         "aud": "www.gusibi.com",
         "sub": key,
         "username": key,
         "scopes": ['open']
    }
    token = jwt.encode(payload, 'secret', algorithm='HS256')
    return True, {'access_token': token, 'account_id': key}
    

def verify_bearer_token(token):
    #  如果在生成token的时候使用了aud参数，那么校验的时候也需要添加此参数
    payload = jwt.decode(token, 'secret', audience='www.gusibi.com', algorithms=['HS256'])
    if payload:
        return True
    return False

@app.route('/v1/apis/session',methods=['POST'])
def make_session(): 
    # 简单生成token,作为某个用户对话的凭证
    result,tokens = create_token(request)
    if result==True:
        return jsonify(tokens)
    return jsonify({"result":status_code['fail']})

@app.route('/v1/apis/reply',methods=['GET','POST'])
def get_reply():
    data = request.get_data().decode('utf-8')
    # try:
    data = json.loads(data)
    # if verify_bearer_token(data['token'])==False:   #短会话不需要一对一持续交流
    #     return jsonify({"result":status_code['invalid_token']})

    # classify intention
    sentence,score = corpus.get_similarity(data['text'])
    print(score,sentence)
    if score<0.1:
        sentence = claw_answer(data['text'])
    return jsonify({'say':sentence})
    # except:
    #     return jsonify({"result":status_code['fail']})
        
app.after_request(after_request)
app.run(host='0.0.0.0', port=config['port'], debug=False)