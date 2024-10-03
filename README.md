# MMAction2 학습 시키기

## Step1. Docker 환경 설정

학습을 위한 환경을 구성하기위해 Docker를 사용합니다.

### Docker 이미지 빌드

프로젝트 디렉토리 내 ```Dockerfile```을 사용하여 도커 이미지를 빌드합니다.
```
docker build -t pls-train:v1.0 .
```

### Docker 컨테이너 실행

아래 명령어를 사용하여 도커 컨테이너를 실행합니다.
```
docker run -d \
  --gpus all \
  -it \
  --name pls \
  pls-train:v1.0
```

만약, 로컬에 있는 동영상 디렉토리를 컨테이너 내 마운트하고 싶으면 아래 명령어를 사용하세요.
```
docker run -d \
  --gpus all \
  -it \
  -v [$로컬 동영상 디렉토리]:/mmaction2/videos \
  --name pls-train \
  pls-train:v1.0
```

### Docker 컨테이너 진입

아래 명령어를 통해 컨테이너에 진입하거나 VSCode 확장 ```Dev Containers```를 사용하여 진입합니다.

```
docker exec -it pls-train /bin/bash
```

## Step2. Python 환경 설정

### mmlab 패키지 설치

```
mim install mmcv==2.1.0
mim install mmdet
mim install mmengine
mim install mmpose
```

위 명령어를 사용하여 mmlab 패키지 설치 후, ```mim list```명령어를 통해 아래와 버젼이 동일한지 확인합니다.
```
Package    Version    Source
---------  ---------  -----------------------------------------
mmcv       2.1.0      https://github.com/open-mmlab/mmcv
mmcv-full  1.3.8      https://github.com/open-mmlab/mmcv
mmdet      3.2.0      https://github.com/open-mmlab/mmdetection
mmengine   0.10.5     https://github.com/open-mmlab/mmengine
mmpose     1.3.2      https://github.com/open-mmlab/mmpose
```

### 기타 패키지 설치
```
pip install -r requirements.txt
```

### 설치 확인 (필요시)

설치를 확인하기 위해 아래 명령어를 실행합니다.
```
python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out.mp4 \
    --config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu60.txt
```

## Step3. 동영상 준비

학습에 사용할 동영상을 준비하여 ```/mmaction2/videos``` 디렉토리 밑에 위치시킵니다. 

만약 ```videos```디렉토리가 없다면 생성해줍니다.

## Step4. Annotation 생성

학습에 사용할 주석 파일을 만들기 위해 아래 과정을 진행합니다.

### Label Map 생성

```data/labelmap.txt```파일을 생성 후, Label목록을 작성합니다.

```
# 예시
normal
selfharm
falldown
```

### Label 생성

아래 명령어를 통해 ```filename,label```로 구성된 csv파일을 생성합니다.
```
python data/generate_label.py
```

- ```label.csv``` : 전체 동영상에 대한 Label 파일입니다.
- ```train.csv``` : 학습에 사용할 동영상에 대한 Label 파일입니다.
- ```val.csv``` : 검증에 사용할 동영상에 대한 Label 파일입니다.

```train```, ```val```의 비율은 8:2로 설정되어 있으나 필요시, ```data/generate_label.py``` 58번째 줄 값을 수정하여 사용하세요.
```
    data/generate_label.py
    ...
58  split_ratio = 0.8 #8:2 비율로 분리
    ...
```

### 동영상 Annotation 생성

아래 명령어를 통해 동영상에 대한 ```.pkl```파일을 생성합니다.
```
python data/generate_anno.py $target --device $device --process $process
```
- ```target``` : 학습, 검증, 미완료파일로 구성됩니다. (입력값 : ```train```, ```val```, ```check``` )
    - ```train``` : 학습에 사용할 동영상에 대한 피클파일을 생성합니다.
    - ```val``` : 검증에 사용할 동영상에 대한 피클파일을 생성합니다.
    - ```check``` : 아직 피클파일로 생성되지 않은 동영상 목록을 출력합니다.
- ```device``` : 사용할 GPU를 구성합니다. (단일 GPU 사용시 : ```'cuda:0'```, 다중 GPU 사용시 : ```'cuda:0,cuda:1,cuda:2,cuda:3'```)
- ```process``` : 하나의 GPU에서 한번에 처리할 동영상 갯수를 설정합니다. (싱글/멀티 GPU 모두 지원)

생성된 파일은 ```data/anno/train```, ```data/anno/val```에 위치합니다.

> 사람이 없는 파일 또는 뼈대 추출 Score가 일정 수준 이하인 경우 피클파일이 생성되지 않을 수 있습니다.

### Annotation 파일 병합

아래 명령어를 통해 개별 동영상에 대한 피클파일을 ```train```, ```val```피클 파일로 병합합니다.
```
python data/combine_anno.py
```
- ```custom_dataset.pkl``` : 전체 동영상에 대한 Annotation 파일입니다.
- ```custom_dataset_train.pkl``` : 학습에 사용할 동영상에 대한 Annotation 파일입니다.
- ```custom_dataset_val.pkl``` : 검증에 사용할 동영상에 대한 Annotation 파일입니다.

## Step5. Config 구성

이미 구성된 ```configs/skeleton/posec3d/config.py```를 사용합니다.
> 필요시 config.py 파일을 수정하여 사용합니다.

## Step6. 학습

### 단일 GPU를 사용하여 학습 진행

```
python tools/train.py configs/skeleton/posec3d/config.py --seed=0 --deterministic
```

### 다중 GPU를 사용하여 학습 진행

```
bash tools/dist_train.sh configs/skeleton/posec3d/config.py ${GPUS} --seed=0 --deterministic
```

학습 진행시 발생하는 가중치 및 로그는 ```work_dirs/config```디렉토리 아래 위치합니다.

## Step7. 데모

아래 명령어를 통해 학습시킨 모델을 실행할 수 있습니다.
```
python demo/demo_skeleton.py ${VIDEO_FILE} ${OUT_FILENAME} \
    [--config ${SKELETON_BASED_ACTION_RECOGNITION_CONFIG_FILE}] \
    [--checkpoint ${SKELETON_BASED_ACTION_RECOGNITION_CHECKPOINT}] \
    [--label-map ${LABEL_MAP}] \
    [--device ${DEVICE}]

# 예시
python demo/demo_skeleton.py demo/demo_custom.mp4 demo/demo_custom_result.mp4
```
