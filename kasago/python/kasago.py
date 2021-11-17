from flask import Flask, json
from flask_cors import CORS

#Flaskオブジェクトを作る
app = Flask(__name__)
CORS(app)
#写真フォルダのパス
DATA_DIR = '/kasago/picture/'
#キーワードリスト
KEYWORDS = {"カサゴ":"kasago","マアジ":"maaji","フウセンウオ":"baloon"}
#ルート
@app.route('/')
def index():
    return 'hello'

@app.route('/kasago/get')
def getKasago():
    #画像をダウンロードする処理
    params = {
            'kudou':'1',
            'onigiri':'1000',
            'weight':'110',
            'sumou':'yokozuna',
    }
    json_str = json.dumps(params, ensure_ascii=False, indent=2) 
    return json_str

@app.route('/kasago/post')
def postKasago():

    #画像を判定する処理
    #カサゴかどうかが95%以上だったら1を返す？違ったら0を返す
    return

app.run(debug=True,host='0.0.0.0')