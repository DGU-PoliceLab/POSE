# 이 파일은 동영상별 피클 파일을 수집하여 하나의 피클 파일로 저장하는 코드입니다.
# 이 파일을 실행하기 전 data/generate_anno.py를 실행하여 동영상별 피클 파일을 생성해야 합니다.
# 사용법 : python combine_anno.py
import os
import pickle

def collect_pickle_files(data_path):
    """지정된 디렉토리에서 모든 피클 파일을 수집하여 리스트로 반환합니다."""
    collected_data = []
    for filename in os.listdir(data_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(data_path, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                collected_data.append(data)  # 피클 파일의 데이터를 리스트에 추가
    return collected_data

def save_data_to_pickle(data, filename):
    """리스트 데이터를 지정된 파일명으로 피클 형식으로 저장합니다."""
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def save_data_to_one_pickle(train, val, filename):
    """리스트 데이터를 지정된 파일명으로 피클 형식으로 저장합니다."""
    data = {'split':{'train': train, 'val': val}}
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    # 훈련 및 검증 데이터가 있는 디렉토리 경로 설정
    train_path = 'data/anno/train'  # 훈련 데이터 디렉토리
    val_path = 'data/anno/val'      # 검증 데이터 디렉토리
    output_path = 'data/anno'

    # 훈련 데이터와 검증 데이터 수집
    train_data = collect_pickle_files(train_path)
    val_data = collect_pickle_files(val_path)

    # 수집한 데이터를 각각의 피클 파일로 저장
    save_data_to_pickle(train_data, f'{output_path}/custom_dataset_train.pkl')
    save_data_to_pickle(val_data, f'{output_path}/custom_dataset_val.pkl')
    save_data_to_one_pickle(train_data, val_data, f'{output_path}/custom_dataset.pkl')

    print("훈련 및 검증 데이터가 성공적으로 저장되었습니다.")