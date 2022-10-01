import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing
from scipy.io import wavfile
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import os
import json

# data 로드 경로
jsonpath = "./data/jsonfile"
wavpath = "./data/wavfile"

# 값을 저장할 리스트
mfccList = []
rapidList = []
loudList = []

# MFCC 객체를 담을 리스트
processMFCCList = []


# MFCC class
class MFCC:
    # 속성 생성
    def __init__(self, name, rapidness, loudness, mfcc):
        self.name = name,
        self.rapidness = rapidness,
        self.loudness = loudness,
        self.mfcc = mfcc


# 폴더에 있는 파일의 개수 반환
def get_files_count(folder_path):
    dirListing = os.listdir(folder_path)
    return len(dirListing)


# fast: 1, medium: 2, low: 3
def classifyRapidness(file_path):
    rapid = 0
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
        # print(json_data['rapidness'])

        # rapidness에 따른 값 부여
        if json_data['rapidness'] == "fast":
            rapid = 1
        elif json_data['rapidness'] == "medium":
            rapid = 2
        else:
            rapid = 3

    return rapid


# very-loud: 1, loud: 2, medium 3, low: 4, very-low: 5
def classifyLoudness(file_path):
    loud = 0
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
        # print(json_data['loudness'])

        # loudness에 따른 값 부여
        if json_data['loudness'] == "very-loud":
            loud = 1
        elif json_data['loudness'] == "loud":
            loud = 2
        elif json_data['loudness'] == "medium":
            loud = 3
        elif json_data['loudness'] == "low":
            loud = 4
        else:
            loud = 5

    return loud


# wav 전처리, mfcc
for i in range(0, get_files_count(wavpath)):
    path = wavpath + "/sit" + str(i + 1) + ".wav"

    sig, sr = librosa.load(path, sr=16000)  # sr = sampling rate

    # mfcc
    mfcc = librosa.feature.mfcc(y=sig, sr=16000, n_mfcc=40, n_fft=400, hop_length=160)

    # mfcc 길이 조정
    # 코드 해체해서 동작 과정을 알고싶지만,, 바쁘기 때문에 다음에 코드 리뷰를 해보겠습니다.
    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
    mfcc = pad2d(mfcc, 80)  # 음성 데이터를 약 1초로 맞춘다. 짧은 경우는 padding(0)을 채움
    mfccList.append(mfcc)

    # 파일 분류
    # 1. rapidness
    rapid = classifyRapidness("./data/jsonfile/sit" + str(i + 1) + ".json")
    rapidList.append(rapid)

    # 2. loudness
    loud = classifyLoudness("./data/jsonfile/sit" + str(i + 1) + ".json")
    loudList.append(loud)


# 스케일링 : 정규화
mfcc_np = np.array(mfccList)
print(mfcc_np.shape)

# 분류기에서 사용하기 위해 3차원 벡터를 2차원 벡터로 변환
mfcc_np = mfcc_np.reshape((100, 40*80))

# 정규화: 추출한 벡터를 0~1 사이의 값으로 변환
scaler = MinMaxScaler()
mfcc_np = scaler.fit_transform(mfcc_np)
print(mfcc_np)
print(len(mfcc_np))

# MFCC object 생성 및 리스트(processMFCCList)에 MFCC 객체 삽입
for i in range(0, get_files_count(wavpath)):
    name = "sit" + str(i + 1) + ".wav"

    # mfcc + 정보(loud, rapid) 담긴 class 생성
    object_mfcc = MFCC(name, rapidList[i], loudList[i], mfcc_np[i])

    # object를 배열에 삽입
    processMFCCList.append(object_mfcc)

