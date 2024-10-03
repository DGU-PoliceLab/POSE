# 이 파일은 특정 디렉토리에 있는 데이터(동영상)에 대해 레이블을 붙이고, train과 val 데이터로 나누는 역할을 합니다.
# 사용법 : python generate_label.py
import os
import csv
import random

def split_data(data, split_ratio=0.8):
    """
    데이터를 주어진 비율로 train과 val로 나누는 함수
    """
    random.shuffle(data)  # 데이터를 랜덤하게 섞음
    split_index = int(len(data) * split_ratio)  # 8:2 비율로 나누기 위한 인덱스 계산
    train_data = data[:split_index]  # train 데이터
    val_data = data[split_index:]    # val 데이터
    return train_data, val_data

def save_to_csv(data, file_path):
    """
    데이터를 CSV 파일로 저장하는 함수
    """
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])  # 헤더 작성
        writer.writerows(data)  # 데이터 작성
    print(f"Data saved to '{file_path}'")

def classify_and_split_files(directory, label_csv, train_csv, val_csv, split_ratio):
    # 파일 분류 데이터를 담을 리스트
    all_data = []

    # 디렉토리 내 파일들을 불러오기
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            # 파일 이름에 'normal'이 포함되면 'normal' 레이블, 그렇지 않으면 'selfharm' 레이블
            if 'normal' in filename.lower():
                label = 'normal'
            else:
                label = 'selfharm'
            
            # 파일 이름과 레이블을 리스트에 추가
            all_data.append([filename, label])

    # label.csv 파일로 저장
    save_to_csv(all_data, label_csv)

    # train과 val 데이터로 나누기
    train_data, val_data = split_data(all_data, split_ratio)

    # train과 val 데이터를 CSV 파일로 저장
    save_to_csv(train_data, train_csv)
    save_to_csv(val_data, val_csv)

if __name__ == '__main__':
    directory_path = '../videos'  # 파일이 있는 디렉토리 경로
    label_csv_path = 'label.csv'    # 저장할 label
    train_csv_path = 'train.csv'       # 저장할 train CSV 파일 경로
    val_csv_path = 'val.csv'           # 저장할 val CSV 파일 경로
    split_ratio = 0.8
    classify_and_split_files(directory_path, label_csv_path, train_csv_path, val_csv_path, split_ratio)