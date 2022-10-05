from flask import Flask, jsonify
from flask_restx import Resource, Api

import os.path
import pickle

## 1. 입력 음성 MFCC 벡터 변환 ##


## 2. MFCC 파일 불러오기 ##
# MFCC 객체 리스트를 저장한 파일
from soundfunc import wav_to_MFCC, classify_similarity, res_cos_sim

file = './MFCC_vector_file.pickle'

if os.path.isfile(file):  # 파일이 존재할 때
    # 파일의 내용 불러오기
    with open(file, 'rb') as file:
        processMFCCList = pickle.load(file)

    print("객체를 불러왔습니다!")


else:  # 파일이 존재하지 않을 때
    # 100개의 sample data에 대한 MFCC 변환
    sample_mfcc = wav_to_MFCC()

    # 파일 생성 및 파일에 객체 데이터 저장하기
    with open(file, "wb") as file:
        pickle.dump(sample_mfcc, file)
    print("객체를 작성했습니다!")

    processMFCCList = sample_mfcc

## 3. 코사인 유사도 계산 ##
cos_sim_list = res_cos_sim(processMFCCList, processMFCCList[77].mfcc) # 입력 음성이 들어갑니다.

## 4. 판정 알고리즘 ##
result = classify_similarity()

print(result)
print("실행 잘 됨!")

## flask server ##
app = Flask(__name__)
api = Api(app, version='0.0.1')
print("서버 실행 좀..")


# test route
@app.route("/")
def index():
    return "Hello World!"


@api.route("/api/result/voice")
class resultVoice(Resource):
    def post(self):
        try:
            # 입력 음성을 어떻게 할려나.? 어떻게 보내주냐에 따라서 코드가 달라질 듯
            # 입력 음성을 노드에서 보내주세요.

            # 음성에 대한 판정 알고리즘 계산, 결과 return 필요
            result = "음성 판정 알고리즘을 작성해주세요. 메서드로 작성해서 return하면 좋을 것 같습니다."

            return jsonify({"success": True, "result": result})  # 판정 결과를 보내준다.
        except:
            return jsonify({"success": False, "message": "사용자 음성 판정 실패"})


if __name__ == '__main__':
    app.run('0.0.0.0', debug=False)

