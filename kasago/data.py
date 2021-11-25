from flask import Flask, json
from icrawler.builtin import BingImageCrawler
import base64

#Flaskオブジェクトを作る
#写真フォルダのパス
DATA_DIR = "kasago/dataset/"
#キーワードリスト
KEYWORDS = {"みかん":"mikan"}
#ラベル
LABEL_DIC = {"kasago":0, "maaji":1}
#収集画像データ数
DATA_COUNT = 120
#画像サイズ
V_SIZE=150
H_SIZE=150
#カテゴリ数
OUT_SIZE=2

#ルート
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
          #キーワードの数だけクローリング
fish = []
for keyword, dirname in KEYWORDS.items():
  fish.append(crawl_image(keyword, dirname, DATA_COUNT, DATA_DIR))
  #画像をダウンロードする処理    
  # json_str = json.dumps(params, ensure_ascii=False, indent=2) 
