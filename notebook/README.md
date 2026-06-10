최종 실행 방법: PowerShell에서 `cd notebook` 후 `python .\notebook_main.py --ip 10.82.166.7` 실행

# Notebook Code

## 실행 방법

Jetson의 현재 IP 주소를 확인한 뒤, 이 폴더에서 아래 명령어를 실행합니다.

```powershell
cd notebook
python .\notebook_main.py --ip 10.82.166.7
```

`10.82.166.7`은 Jetson IP 주소입니다. 연결할 때 Jetson IP가 바뀌면 `--ip` 뒤의 값만 바꾸면 됩니다.

## 라이브러리 설치

Python 패키지는 Pillow만 추가로 필요합니다.

```powershell
pip install pillow
```

영상 수신 기능을 쓰려면 Windows에 GStreamer도 설치되어 있어야 합니다. 설치 후 `gst-launch-1.0` 명령이 PATH에서 실행되면 가장 좋습니다.

```powershell
gst-launch-1.0 --version
```

## 정리 요약

- `notebook_main.py`가 최종 노트북 구동 코드입니다.
- 페달/핸들 입력 수집, Jetson UDP 송신, Jetson 보정값 수신, 영상 스트림 수신, 대시보드 표시 기능을 한 파일에 통합했습니다.
- `01_`부터 `05_`까지의 파일은 개발 과정에서 기능별로 나누어 만들고 테스트했던 코드입니다.
- 현재 `notebook` 폴더는 Python 코드 파일과 이 README만 남겨둔 상태입니다.

## 개발 흐름

- `01_pedal_udp_dashboard.py`

  페달과 핸들 입력을 읽고 Jetson으로 UDP 전송하는 기본 대시보드 코드입니다. 보정된 페달값을 다시 UDP로 받아 그래프에 표시하는 기능도 이 단계에서 만들었습니다.

- `02_receive_jetson_video.py`

  Jetson에서 보내는 RTP/JPEG UDP 영상 스트림을 Windows 노트북에서 GStreamer로 수신하는 테스트 코드입니다.

- `03_drive_with_video_original.py`

  페달/핸들 UDP 송신 코드와 Jetson 영상 수신 코드를 하나의 실행 흐름으로 묶었던 기존 통합 코드입니다.

- `04_detect_wheel_buttons.py`

  휠 버튼 번호와 입력 상태를 확인하기 위한 테스트 코드입니다. 모드 변경 버튼과 비상 버튼을 정하기 위해 사용했습니다.

- `05_wheel_button_state_test.py`

  휠 버튼 눌림 상태 변화를 더 자세히 확인하기 위한 테스트 코드입니다.

- `notebook_main.py`

  최종 통합 코드입니다. 위 개발 단계에서 나뉘어 있던 기능을 한 파일에 합쳐, 이 파일 하나만으로 노트북 측 최종 실행이 가능하도록 구성했습니다.
