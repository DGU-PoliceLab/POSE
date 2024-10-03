# 이 파일은 동영상 파일에서 skeleton 데이터를 추출하여 annotation 파일을 생성하는 스크립트입니다.
# 이 파일을 실행하기 전 generate_label.py를 실행하여 label.csv. train.csv, val.csv 파일을 생성해야 합니다.
# 사용법: python data/generate_anno.py [train or val] --device 'cuda:0, cuda:1, cuda:2, cuda:3'
import os
import argparse
import subprocess
from multiprocessing import Process
import csv

def read_label_csv(file_path):
    """
    CSV 파일을 읽어서 데이터를 반환하는 함수
    """
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def check_anno_done(anno_path):
    """지정된 디렉토리에 있는 파일명의 리스트를 반환합니다 (확장자 제외)."""
    anno_list = []
    
    # 디렉토리 내의 모든 파일 목록 가져오기
    for entry in os.listdir(anno_path):
        # 파일인지 확인
        if os.path.isfile(os.path.join(anno_path, entry)):
            # 확장자 제외하고 파일명 추가
            file_name, _ = os.path.splitext(entry)
            anno_list.append(file_name)
    
    return anno_list

def check_anno_exists(file_path):
    if os.path.exists(file_path):
        print(f"annotation 파일({file_path})이 존재합니다. 생성을 건너뜁니다.")
        return True
    else:
        return False
    
def create_work_list(done_list, label_data):
    work_list = []
    for filename, label in label_data[1:]:
        if filename[:-4] not in done_list:
            work_list.append((filename, label))
    return work_list

def create_annotation_file(video_path, output_path, device):
    """
    동영상 파일에서 skeleton 데이터를 추출하여 annotation 파일을 생성하는 함수
    """
    if check_anno_exists(output_path):
        return
    command = f"python data/pose_extraction.py --device '{device}' {video_path} {output_path}"
    subprocess.call(command, shell=True)

def dist_create_annotation_file(video_path_list, output_path_list, device):
    """
    동영상 파일에서 skeleton 데이터를 추출하여 annotation 파일을 생성하는 함수 (멀티 GPU 사용시)
    """
    proc = []
    for i in range(len(device)):
        try:
            p = Process(target=create_annotation_file, args=(video_path_list[i], output_path_list[i], device[i]))
            proc.append(p)
            p.start()
        except:
            pass

    for p in proc:
        p.join()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Skeleton Annotation')
    parser.add_argument('target', type=str, help='train, val')
    parser.add_argument('--device', type=str, default='cuda:0', help='여러대의 GPU를 사용할때, cuda:0,cuda:1,cuda:2,cuda:3 과 같이 입력') # 동일한 GPU에서 여러 프로세스를 실행할 경우, cuda:0,cuda:1,cuda:0,cuda:1 과 같이 입력
    parser.add_argument('--process', type=int, default='1', help='하나의 GPU에서 동시에 처리할 동영상의 갯수')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    global_args = parse_args()
    
    target = global_args.target # annotation 파일 생성 대상
    device = global_args.device # 사용할 GPU 번호
    process = global_args.process # GPU에서 동시에 처리할 동영상의 갯수
    dist = False

    if target == 'check':
        train_lable_data = read_label_csv('data/train.csv')
        val_lable_data = read_label_csv('data/val.csv')
        train_done_list = check_anno_done('data/anno/train')
        val_done_list = check_anno_done('data/anno/val')
        train_work_list = create_work_list(train_done_list, train_lable_data[1:])
        val_work_list = create_work_list(val_done_list, val_lable_data[1:])
        print(f'train: {len(train_work_list)}개의 annotation 파일 생성 필요')
        print(train_work_list)
        print(f'val: {len(val_work_list)}개의 annotation 파일 생성 필요')
        print(val_work_list)
        exit()

    data_dir = 'videos' # 데이터 디렉토리 경로
    label_csv_path = f'data/{target}.csv' # csv 파일 경로
    label_data = read_label_csv(label_csv_path) # csv 파일 읽기
    done_list = check_anno_done(f'data/anno/{target}') # 이미 생성된 annotation 파일 리스트
    work_list = create_work_list(done_list, label_data[1:]) # 작업할 annotation 파일 리스트 생성
    print(f'{target}: {len(work_list)}개의 annotation 파일 생성 필요', work_list)

    if ',' in device:
        device = device.split(',')
        device = device * process
        dist = True

    cur = 1
    last = len(work_list)
    
    # csv 파일의 데이터를 이용하여 annotation 파일 생성
    if dist:
        for start_idx in range(0, len(work_list), len(device)):
            per = cur / last * 100
            video_path_list = []
            output_path_list = []
            for i in range(len(device)):
                if start_idx + i >= len(work_list):
                    break
                filename, label = work_list[start_idx + i]
                print(f'\033[1;32m[{per:.1f}%][{cur}/{last}] - {filename}\033[0m')
                video_path = os.path.join(data_dir, filename)
                output_path = f'data/anno/{target}/{filename[:-4]}.pkl'
                video_path_list.append(video_path)
                output_path_list.append(output_path)
                cur += 1
            dist_create_annotation_file(video_path_list, output_path_list, device)
            
    else:
        for filename, label in work_list:
            per = cur / last * 100
            print(f'\033[1;32m[{per:.1f}%][{cur}/{last}] - {filename}\033[0m')
            video_path = os.path.join(data_dir, filename)
            output_path = f'data/anno/{target}/{filename[:-4]}.pkl'
            create_annotation_file(video_path, output_path, device)
            cur += 1