from flask import Flask, json
from flask_cors import CORS
from icrawler.builtin import BingImageCrawler
import base64
import random

#Flaskオブジェクトを作る
app = Flask(__name__)
CORS(app)
#写真フォルダのパス
DATA_DIR = 'kasago/picture/'
#キーワードリスト
KEYWORDS = {"カサゴ":"kasago","マアジ":"maaji","フウセンウオ":"baloon","マンボウ":"manbo","ヒラメ":"hirame","安倍晋三":"penguin"}
#ラベル
LABEL_DIC = {"kasago":0, "maaji":1,"baloon":2,"manbo":3,"hirame":4,"SYAKE":5}
#収集画像データ数
DATA_COUNT = 1
#画像サイズ
V_SIZE=150
H_SIZE=150
#カテゴリ数
OUT_SIZE=6

#ルート
@app.route('/')
def index():
    return 'hello'

def crawl_image(fishname, dirname, datacount, root_dir):
    crawler = BingImageCrawler(storage={'root_dir':root_dir + dirname})
    #クローリングの実行
    crawler.crawl(
        keyword=fishname,
        max_num=datacount
    )
    file_data = open(root_dir + dirname + '/000001.jpg', "rb").read()
    img_byte = base64.b64encode(file_data).decode("utf-8")
    return img_byte

@app.route('/kasago/get')
def getKasago():
    #キーワードの数だけクローリング
    fish = []
    for keyword, dirname in KEYWORDS.items():
        fish.append(crawl_image(keyword, dirname, DATA_COUNT, DATA_DIR))
    #画像をダウンロードする処理
    #画像をbase64で返す処理
    params = {}
    random.shuffle(fish)
    params["fish"] = fish
    
    json_str = json.dumps(params, ensure_ascii=False, indent=2)
    return json_str

@app.route('/kasago/post')
def postKasago():
    #画像を判定する処理
    #カサゴかどうかが95%以上だったら1を返す？違ったら0を返す
    return

app.run(debug=True,host='0.0.0.0')