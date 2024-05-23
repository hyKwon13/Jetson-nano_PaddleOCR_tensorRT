# 프로젝트 개요
Jetson Nano에서 PaddleOCR을 GPU와 TensorRT를 사용하여 실행하는 방법에 대해 다룹니다. Python 3.6과 3.8에서는 paddlepaddle과 paddleocr 간의 호환성 문제가 있어 Python 3.7을 사용하였습니다.

## Jetpack SDK 설치(간략 설명)
1. **SD 카드 포맷**: SD Card Formatter를 사용하여 64GB microSD 카드를 포맷합니다.
    - [SD Card Formatter 다운로드](https://www.sdcard.org/downloads/formatter/sd-memory-card-formatter-for-windows-download/)
    - 포맷 버튼을 클릭하여 포맷을 완료합니다.
    - ![이미지](https://github.com/hyKwon13/Jetson-nano_PaddleOCR_CUDA/assets/117807382/40fec450-6ca2-48fe-b878-b73b55925ef2)

2. **Jetpack SDK 설치**:
    - [Jetpack SDK 다운로드](https://developer.nvidia.com/jetpack-sdk-464)

3. **balenaEtcher 사용**: balenaEtcher 프로그램을 사용하여 microSD 카드에 이미지 설치.
    - [Etcher 다운로드](https://etcher.balena.io/)

## Python 3.7 설치
Python 3.6과 3.8에서는 paddlepaddle-gpu가 정상적으로 설치되나, paddleocr과의 호환성 문제로 Python 3.7을 설치하였습니다.

1. **Python 3.7 설치**
    ```bash
    sudo apt update
    sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.7
    ```

2. **Python 버전 관리**
    여러 Python 버전을 쉽게 전환할 수 있도록 update-alternatives를 설정합니다.
    ```bash
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
    sudo update-alternatives --config python3
    ```

3. **pip 설치**
    ```bash
    sudo apt install python3.7-distutils
    wget https://bootstrap.pypa.io/get-pip.py
    sudo python3.7 get-pip.py
    ```

4. **가상 환경 생성**
    ```bash
    sudo apt update
    sudo apt install python3.7-venv
    python3.7 -m venv myenv
    ```

5. **가상 환경 활성화**
    ```bash
    source myenv/bin/activate
    ```

## Paddlepaddle-GPU 설치
Python 3.7에 paddlepaddle-gpu를 설치하기 위해 whl 파일을 직접 다운로드하여 설치하였습니다.

1. **paddlepaddle-gpu 다운로드**: [다운로드 링크](https://forums.developer.nvidia.com/t/paddlepaddle-for-jetson/242765)
    - ![이미지](https://github.com/hyKwon13/Jetson-nano_PaddleOCR_CUDA/assets/117807382/fd80b418-f0a8-4eb9-ad90-5e9ecf788fbb)


2. **whl 파일 전송**
   데스크탑에서 ssh를 통해 jetson nano를 사용중이므로 cmd에서 pscp 명령어를 사용하여 whl 파일을 jetson nano로 전송합니다.
   ```bash
    pscp C:\temp\* root@192.168.2.222:/temp
    ```

4. **whl 파일 설치**
    ```bash
    python3.7 -m pip install paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl
    ```

5. **오류 해결**
    ```bash
    Command "python setup.py egg_info" failed with error code 1 in /tmp/pip-build-apmufue2/paddle-bfloat/
    ```

    이 오류는 paddle-bfloat 패키지를 설치하는 과정에서 발생한 문제로, 주로 필요한 빌드 의존성 패키지가 누락되었거나, 설치 과정에서 문제가 발생했을 때 나타납니다. 아래의 방법들을 통해 문제를 해결할 수 있습니다.

    1. **pip 및 setuptools 업그레이드**
        ```bash
        pip3 install --upgrade pip setuptools
        ```

    2. **numpy 미리 설치**
        ```bash
        pip3 install numpy
        ```

    3. **paddle-bfloat 패키지 직접 설치**
        ```bash
        pip install paddle-bfloat==0.1.7
        pip install paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl
        ```

## PaddleOCR 설치

1. **paddleocr 설치**
    ```bash
    pip3 install paddleocr
    ```

2. **오류 해결**
    ```bash
    Failed to build psutil
    ERROR: Could not build wheels for psutil, which is required to install pyproject.toml-based projects
    ```

    이 오류는 psutil 패키지를 빌드하는 과정에서 Python 헤더 파일을 찾을 수 없어서 발생한 문제입니다. 이를 해결하려면 필요한 Python 헤더 파일을 설치해야 합니다.

    다음 명령어를 통해 Python 헤더 파일과 빌드 도구를 설치해보세요.
    ```bash
    sudo apt-get install gcc python3-dev
    pip3 install paddleocr
    ```

    문제가 해결되지 않으면 psutil 패키지를 빌드할 때 필요한 추가적인 의존성을 설치해보세요.
    ```bash
    sudo apt-get install build-essential
    pip install paddleocr
    ```

3. **여전히 오류 발생 시**

    python3-dev 패키지가 누락되었을 가능성이 있으므로, 해당 패키지를 다시 설치해보세요.
    ```bash
    sudo apt-get install python3.7-dev
    pip install psutil
    ```

4. **환경 변수 설정 확인**
    Python 헤더 파일을 찾지 못하는 경우, 환경 변수가 제대로 설정되지 않았을 수 있습니다. 아래 명령어를 사용하여 CPATH 환경 변수를 설정해 보세요.
    ```bash
    export CPATH=/usr/include/python3.7m
    ```

    다시 psutil 설치
    ```bash
    pip install psutil
    ```

5. **OpenCV 환경 변수 설정**
    ```bash
    ImportError: OpenCV loader: missing configuration file: ['config-3.7.py', 'config-3.py']. Check OpenCV installation.
    ```

    이 오류는 OpenCV가 설치된 경로가 올바르게 설정되어 있는지 확인하세요. 경우에 따라 환경 변수를 수동으로 설정해야 할 수도 있습니다. Python 3.7로 환경 변수를 설정합니다.
    ```bash
    export PYTHONPATH=~/project/paddle3.7/lib/python3.7/site-packages:$PYTHONPATH
    ```

## 설치 확인 예제
다음 예제 코드를 통해 PaddleOCR이 GPU와 TensorRT를 제대로 사용하고 있는지 확인할 수 있습니다. 이를 통해 onnxruntime보다 더 빠른 추론 속도를 얻을 수 있습니다. 다만, TensorRT를 사용할 경우 초기 구동 시 많은 메모리를 사용하여 시작 속도가 느려질 수 있습니다.

```bash
import cv2
from paddleocr import PaddleOCR

def draw_rectangle(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt, rectangles
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)
        bottom_right_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            bottom_right_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        rectangles.append((top_left_pt, bottom_right_pt))

def perform_ocr(frame, top_left_pt, bottom_right_pt, ocr):
    if top_left_pt == (-1, -1) or bottom_right_pt == (-1, -1):
        return
    roi = frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
    if roi.size == 0:
        return
    result = ocr.ocr(roi, cls=False)
    result = result[0]
    if result is not None:
        for line in result:
            print(line[1][0])
            cv2.putText(frame, line[1][0], (top_left_pt[0], bottom_right_pt[1]+21), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.namedWindow('paddleocr')
cv2.setMouseCallback('paddleocr', draw_rectangle)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
rectangles = [] 

ocr = PaddleOCR(
    use_gpu=True,
    use_tensorrt=True,
    use_angle_cls=False,
    lang='en',
    show_log=False,
)


            
for _ in range(30):
    _, _ = cap.read()

while True:
    ret, frame = cap.read()

    for rect in rectangles:
        cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 2)
    
    if drawing and top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
        cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
    if not drawing and top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
        perform_ocr(frame, top_left_pt, bottom_right_pt, ocr)
    cv2.imshow('paddleocr', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```
