from flask import Flask, json,request
from flask_cors import CORS
from icrawler.builtin import BingImageCrawler
import base64
import random
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import os
import glob
import random
import shutil
import sys
import cv2
import numpy as np
from flask import Flask
from keras.models import load_model
import keras
from keras.utils import np_utils
from icrawler.builtin import BingImageCrawler
from keras.preprocessing.image import array_to_img,img_to_array,load_img
from sklearn.model_selection import train_test_split

#Flaskオブジェクトを作る
app = Flask(__name__)
CORS(app)
#写真フォルダのパス
DATA_DIR = 'kasago/picture/'
#キーワードリスト
KEYWORDS = {"カサゴ":"kasago","マアジ":"maaji","フウセンウオ":"baloon","マンボウ":"manbo","ヒラメ":"hirame","安倍晋三":"penguin"}
#ラベル
LABEL_DIC = {"kasago":0, "maaji":1}
#収集画像データ数
DATA_COUNT = 1
#画像サイズ
V_SIZE=150
H_SIZE=150
#カテゴリ数
OUT_SIZE=2

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

def reshape_numpy(imagefile, imgsize, ch):
        """画像のデータ化
        指定した画像のサイズを変更し、ndarrayに変換する
        Args：
            imagefile(str):画像のパス
            imgsize(tuple(int,int)):リサイズ後の画像サイズ
            ch(int):画像の色空間
        Returns:
            X_train: numpy.ndarray 教師データ
            y_train: numpy.ndarray ラベル
        """
        temp_img = load_img(imagefile, target_size=(imgsize, imgsize))
        temp_img_array = img_to_array(temp_img)
        temp_img_array = temp_img_array.astype('float32')/255.0
        temp_img_array = temp_img_array.reshape((1, imgsize, imgsize, ch))
        return temp_img_array

def get_keys_from_value(d,val):
        return[k for k, v in d.items() if v == val]

def resize(path,v,h):

  img = cv2.imread(path)
  img = cv2.resize(img, (v,h))
  return img

def bundle_resize(dirname):
  """フォルダ内画像のリサイズ
      指定したフォルダ内の画像をリサイズ
      Args：
          dirname(str): フォルダパス
  """
  path_list = glob.glob(dirname + "/*")
  for path in path_list:
    path = path.replace('\\','/')
    img = resize(path,V_SIZE,H_SIZE)
    cv2.imwrite(path,img)
  return

def move_file(dirname, dist_path, no):
  """ファイルの移動
      ランダムにファイルを選択し、指定のディレクトリに移動する
      Args；
          dirname(str):移動元ファイルのしれく鳥
          dist_path(str):保存先のディレクトリ
          no(int):ファイル名(処理回数)
  """
  path_list=glob.glob(dirname + '/*')
  test_file = random.choice(path_list)
  test_file = test_file.replace('\\','/')
  shutil.move(test_file, dist_path+str(no)+".jpg")
  
def create_datasets(dirname):
    X_train=[]
    y_train=[]
    data=os.listdir(dirname)
    for row in data:
        if row == 'test':
            continue
        i=os.listdir(dirname)
        for target_file in i:
            image=(dirname+"/"+target_file)
            temp_img=load_img(image)
            temp_img_array=img_to_array(temp_img)
            X_train.append(temp_img_array)
            y_train.append(LABEL_DIC["kasago"])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train,y_train


#画像をダウンロードする処理
#base64で返す
@app.route('/kasago/get')
def getKasago():
    #キーワードの数だけクローリング
    fish = []
    for keyword, dirname in KEYWORDS.items():
        fish.append(crawl_image(keyword, dirname, DATA_COUNT, DATA_DIR))
    params = {}
    random.shuffle(fish)
    params["fish"] = fish
    json_str = json.dumps(params, ensure_ascii=False, indent=2)
    return json_str
#カサゴを判定する処理
@app.route('/kasago/answer/get',methods=["POST"])
def postKasago():
    print(request.form)
    #確認テストファイル名
    image_file=r"decode.jpg"
    img_binary = base64.b64decode(b)
    jpg=np.frombuffer(img_binary,dtype=np.uint8)
    #raw image <- jpg
    img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    #画像を保存する場合
    cv2.imwrite(image_file,img)

    #画像判定
    test_imagefile_path = 'kasago/picture/' + image_file
    img_array = reshape_numpy(test_imagefile_path, 150,3)
    model = load_model(DATA_DIR + 'kasagoLearn.h5')
    probs = model.predict(img_array)

    #画像の表示
    img = cv2.imread(test_imagefile_path)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()

    label = get_keys_from_value(LABEL_DIC, np.argmax(probs[0]))
    print("AI判定画像は",probs[0][np.argmax(probs[0])]*100,"%の確立で",label,"です")
    params = {}
    if probs[0][np.argmax(probs[0])]*100 > 95:
        params["answer"] = "1"
    else:
        params["answer"] = "0"
    json_str = json.dumps(params, ensure_ascii=False, indent=2)
    return json_str



# #メイン処理 学習モデルが大きすぎるからサーバ起動時に学習モデルを作成
# #画像が格納されたフォルダ内の画像全てをリサイズ
# #データセットの作成
# X_train,y_train = create_datasets("kasago/dataset/")
# #学習用とテスト用に分割する
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, train_size = 0.8)
# #データの正規化
# X_train = X_train.astype('float32') / 255
# X_test = X_test.astype('float32') / 255
# #ラベルデータをOne-Hotベクトルに直す
# y_train = keras.utils.np_utils.to_categorical(y_train.astype('int32'),OUT_SIZE)
# y_test = keras.utils.np_utils.to_categorical(y_test.astype('int32'),OUT_SIZE)

# #入力と出力を指定
# im_rows = 150 #画像の縦ピクセルサイズ
# im_cols = 150 #画像の横のピクセルサイズ
# im_color = 3 #画像の色空間 / RGBカラー
# in_shape = (im_rows, im_cols, im_color)
# out_size = 2 #分類数
# epochs = 1 #学習回数

# #MLPモデルを定義
# model = Sequential()
# model.add(Conv2D(32,kernel_size=(3,3),
#                  activation='relu',
#                  input_shape=in_shape))
# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(out_size,activation='softmax'))

# #モデルを構築
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=RMSprop(),
#     metrics=['accuracy']
# )

# #学習
# hist = model.fit(X_train,y_train,
#                  batch_size=128,
#                  epochs=epochs,
#                  verbose=1,
#                  validation_data=(X_test,y_test))

# model.save(DATA_DIR + 'kasagoLearn.h5')
# print("saved")
app.run(debug=True,host='0.0.0.0')from flask import Flask, json,request
from flask_cors import CORS
from icrawler.builtin import BingImageCrawler
import base64
import random
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import os
import glob
import random
import shutil
import sys
import cv2
import numpy as np
from flask import Flask
from keras.models import load_model
import keras
from keras.utils import np_utils
from icrawler.builtin import BingImageCrawler
from keras.preprocessing.image import array_to_img,img_to_array,load_img
from sklearn.model_selection import train_test_split

#Flaskオブジェクトを作る
app = Flask(__name__)
CORS(app)
#写真フォルダのパス
DATA_DIR = 'kasago/picture/'
#キーワードリスト
KEYWORDS = {"カサゴ":"kasago","マアジ":"maaji","フウセンウオ":"baloon","マンボウ":"manbo","ヒラメ":"hirame","安倍晋三":"penguin"}
#ラベル
LABEL_DIC = {"kasago":0, "maaji":1}
#収集画像データ数
DATA_COUNT = 1
#画像サイズ
V_SIZE=150
H_SIZE=150
#カテゴリ数
OUT_SIZE=2

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

def reshape_numpy(imagefile, imgsize, ch):
        """画像のデータ化
        指定した画像のサイズを変更し、ndarrayに変換する
        Args：
            imagefile(str):画像のパス
            imgsize(tuple(int,int)):リサイズ後の画像サイズ
            ch(int):画像の色空間
        Returns:
            X_train: numpy.ndarray 教師データ
            y_train: numpy.ndarray ラベル
        """
        temp_img = load_img(imagefile, target_size=(imgsize, imgsize))
        temp_img_array = img_to_array(temp_img)
        temp_img_array = temp_img_array.astype('float32')/255.0
        temp_img_array = temp_img_array.reshape((1, imgsize, imgsize, ch))
        return temp_img_array

def get_keys_from_value(d,val):
        return[k for k, v in d.items() if v == val]

def resize(path,v,h):

  img = cv2.imread(path)
  img = cv2.resize(img, (v,h))
  return img

def bundle_resize(dirname):
  """フォルダ内画像のリサイズ
      指定したフォルダ内の画像をリサイズ
      Args：
          dirname(str): フォルダパス
  """
  path_list = glob.glob(dirname + "/*")
  for path in path_list:
    path = path.replace('\\','/')
    img = resize(path,V_SIZE,H_SIZE)
    cv2.imwrite(path,img)
  return

def move_file(dirname, dist_path, no):
  """ファイルの移動
      ランダムにファイルを選択し、指定のディレクトリに移動する
      Args；
          dirname(str):移動元ファイルのしれく鳥
          dist_path(str):保存先のディレクトリ
          no(int):ファイル名(処理回数)
  """
  path_list=glob.glob(dirname + '/*')
  test_file = random.choice(path_list)
  test_file = test_file.replace('\\','/')
  shutil.move(test_file, dist_path+str(no)+".jpg")
  
def create_datasets(dirname):
    X_train=[]
    y_train=[]
    data=os.listdir(dirname)
    for row in data:
        if row == 'test':
            continue
        i=os.listdir(dirname)
        for target_file in i:
            image=(dirname+"/"+target_file)
            temp_img=load_img(image)
            temp_img_array=img_to_array(temp_img)
            X_train.append(temp_img_array)
            y_train.append(LABEL_DIC["kasago"])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train,y_train


#画像をダウンロードする処理
#base64で返す
@app.route('/kasago/get')
def getKasago():
    #キーワードの数だけクローリング
    fish = []
    for keyword, dirname in KEYWORDS.items():
        fish.append(crawl_image(keyword, dirname, DATA_COUNT, DATA_DIR))
    params = {}
    random.shuffle(fish)
    params["fish"] = fish
    json_str = json.dumps(params, ensure_ascii=False, indent=2)
    return json_str
#カサゴを判定する処理
@app.route('/kasago/answer/get',methods=["POST"])
def postKasago():
    print(request.form)
    #確認テストファイル名
    image_file=r"decode.jpg"
    img_binary = base64.b64decode(b)
    jpg=np.frombuffer(img_binary,dtype=np.uint8)
    #raw image <- jpg
    img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    #画像を保存する場合
    cv2.imwrite(image_file,img)

    #画像判定
    test_imagefile_path = 'kasago/picture/' + image_file
    img_array = reshape_numpy(test_imagefile_path, 150,3)
    model = load_model(DATA_DIR + 'kasagoLearn.h5')
    probs = model.predict(img_array)

    #画像の表示
    img = cv2.imread(test_imagefile_path)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()

    label = get_keys_from_value(LABEL_DIC, np.argmax(probs[0]))
    print("AI判定画像は",probs[0][np.argmax(probs[0])]*100,"%の確立で",label,"です")
    params = {}
    if probs[0][np.argmax(probs[0])]*100 > 95:
        params["answer"] = "1"
    else:
        params["answer"] = "0"
    json_str = json.dumps(params, ensure_ascii=False, indent=2)
    return json_str



# #メイン処理 学習モデルが大きすぎるからサーバ起動時に学習モデルを作成
# #画像が格納されたフォルダ内の画像全てをリサイズ
# #データセットの作成
# X_train,y_train = create_datasets("kasago/dataset/")
# #学習用とテスト用に分割する
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, train_size = 0.8)
# #データの正規化
# X_train = X_train.astype('float32') / 255
# X_test = X_test.astype('float32') / 255
# #ラベルデータをOne-Hotベクトルに直す
# y_train = keras.utils.np_utils.to_categorical(y_train.astype('int32'),OUT_SIZE)
# y_test = keras.utils.np_utils.to_categorical(y_test.astype('int32'),OUT_SIZE)

# #入力と出力を指定
# im_rows = 150 #画像の縦ピクセルサイズ
# im_cols = 150 #画像の横のピクセルサイズ
# im_color = 3 #画像の色空間 / RGBカラー
# in_shape = (im_rows, im_cols, im_color)
# out_size = 2 #分類数
# epochs = 1 #学習回数

# #MLPモデルを定義
# model = Sequential()
# model.add(Conv2D(32,kernel_size=(3,3),
#                  activation='relu',
#                  input_shape=in_shape))
# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(out_size,activation='softmax'))

# #モデルを構築
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=RMSprop(),
#     metrics=['accuracy']
# )

# #学習
# hist = model.fit(X_train,y_train,
#                  batch_size=128,
#                  epochs=epochs,
#                  verbose=1,
#                  validation_data=(X_test,y_test))

# model.save(DATA_DIR + 'kasagoLearn.h5')
# print("saved")
app.run(debug=True,host='0.0.0.0')