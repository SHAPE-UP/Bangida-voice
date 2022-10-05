import json
import os
import os.path
import pickle

import librosa.display
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import dot
from numpy.linalg import norm

# data 로드 경로
jsonpath = "./data/jsonfile"
wavpath = "./data/wavfile"

# 값을 저장할 리스트
mfccList = []
rapidList = []
loudList = []

# MFCC 객체를 담을 리스트
processMFCCList = []

# 배열 선언
voiceSimilarity = []  # 유사도를 담은 배열
processSimilarityList = []  # VOICESIM 객체를 담은 배열

# 유사도 분석에서 사용하는 배열
rapidness_cos_avg_list = []
loudness_cos_avg_list = []


# MFCC class
class MFCC:
    def __init__(self, name, rapidness, loudness, mfcc):
        self.name = name
        self.rapidness = rapidness
        self.loudness = loudness
        self.mfcc = mfcc


# VOICESIM class
class VOICESIM:
    def __init__(self, name, rapidness, loudness, similarity):
        self.name = name
        self.rapidness = rapidness
        self.loudness = loudness
        self.similarity = similarity


# 폴더에 있는 파일의 개수 반환
def get_files_count(folder_path):
    dirListing = os.listdir(folder_path)
    return len(dirListing)


# fast: 3, medium: 2, low: 1
def classify_rapidness(file_path):
    rapid = 0
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
        # print(json_data['rapidness'])

        # rapidness에 따른 값 부여
        if json_data['rapidness'] == "fast":
            rapid = 3
        elif json_data['rapidness'] == "medium":
            rapid = 2
        else:
            rapid = 1

    return rapid


# very-loud: 5, loud: 4, medium 3, low: 2, very-low: 1
def classify_loudness(file_path):
    loud = 0
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
        # print(json_data['loudness'])

        # loudness에 따른 값 부여
        if json_data['loudness'] == "very-loud":
            loud = 5
        elif json_data['loudness'] == "loud":
            loud = 4
        elif json_data['loudness'] == "medium":
            loud = 3
        elif json_data['loudness'] == "low":
            loud = 2
        else:
            loud = 1

    return loud


# .wav -> MFCC 벡터로 변환
def wav_to_MFCC():
    # wav 전처리, mfcc
    for i in range(0, get_files_count(wavpath)):
        path = wavpath + "/sit" + str(i + 1) + ".wav"

        sig, sr = librosa.load(path, sr=16000)  # sr = sampling rate

        # mfcc
        mfcc = librosa.feature.mfcc(y=sig, sr=16000, n_mfcc=40, n_fft=400, hop_length=160)

        # mfcc 길이 조정
        # 코드 해체해서 동작 과정을 알고싶지만,, 바쁘기 때문에 다음에 코드 리뷰를 해보겠습니다.
        pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
        mfcc = pad2d(mfcc, 80)  # 음성 데이터를 약 1초로 맞춘다. 1초보다 짧은 경우는 padding(0)을 채움
        mfccList.append(mfcc)

        # 파일 분류
        # 1. rapidness
        rapid = classify_rapidness("./data/jsonfile/sit" + str(i + 1) + ".json")
        rapidList.append(rapid)

        # 2. loudness
        loud = classify_loudness("./data/jsonfile/sit" + str(i + 1) + ".json")
        loudList.append(loud)

    # 스케일링 : 정규화
    mfcc_np = np.array(mfccList)

    # 분류기에서 사용하기 위해 3차원 벡터를 2차원 벡터로 변환
    mfcc_np = mfcc_np.reshape((100, 40 * 80))

    # 정규화: 추출한 벡터를 0~1 사이의 값으로 변환
    scaler = MinMaxScaler()
    mfcc_np = scaler.fit_transform(mfcc_np)

    # MFCC object 생성 및 리스트(processMFCCList)에 MFCC 객체 삽입
    for i in range(0, get_files_count(wavpath)):
        name = "sit" + str(i + 1) + ".wav"

        # mfcc + 정보(loud, rapid) 담긴 class 생성
        object_mfcc = MFCC(name, rapidList[i], loudList[i], mfcc_np[i])

        # object를 배열에 삽입
        processMFCCList.append(object_mfcc)

    return processMFCCList


# 코사인 유사도 함수
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


# 입력 음성과 샘플링 음성의 코사인 유사도를 구함
def res_cos_sim(mfcc_list, input_sound):  # 샘플링 음성의 mfcc 리스트, 입력 음성의 벡터
    # 유사도 구하는 반복문
    for i in range(0, len(mfcc_list)):
        # 입력 음성과 100개의 샘플 음성 코사인 유사도 계산

        voiceSimilarity.append(cosine_similarity(mfcc_list[i].mfcc, input_sound))

    # VOICESIM 객체 삽입
    for i in range(0, 100):
        name = "sit" + str(i + 1) + ".wav"

        # 코사인 유사도 + 정보(loud, rapid) 담긴 class 생성
        object_cosine = VOICESIM(name, mfcc_list[i].rapidness, mfcc_list[i].loudness, voiceSimilarity[i])

        # object를 배열에 삽입
        processSimilarityList.append(object_cosine)

    return processSimilarityList


# 유사도를 비교하여 분류하는 작업
def classify_similarity():
    # 각 그룹에서 유사도 비교

    for i in range(1, 4, 1):
        ll = [item.similarity for item in processSimilarityList if item.rapidness == i]
        avg = sum(ll) / len(ll)
        rapidness_cos_avg_list.append(avg)
    input_rapidness = rapidness_cos_avg_list.index(max(rapidness_cos_avg_list)) + 1

    for i in range(1, 6, 1):
        ll = [item.similarity for item in processSimilarityList if item.loudness == i]
        avg = sum(ll) / len(ll)
        loudness_cos_avg_list.append(avg)
    input_loudness = loudness_cos_avg_list.index(max(loudness_cos_avg_list)) + 1

    # print("입력 음성의 빠르기, 크기", input_rapidness, input_loudness)
    classify_result = [input_rapidness, input_loudness]

    return classify_result
