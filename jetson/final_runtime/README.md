최종 실행 명령어: `MPLCONFIGDIR=/tmp/matplotlib python3 jetson_main.py --serial-port /dev/ttyUSB0 --video-udp-host <NOTEBOOK_IP>`

# Jetson Runtime

## 최종 실행 방법

1. Arduino serial 권한을 열어줍니다.

```bash
sudo chmod a+rw /dev/ttyUSB0
```

2. Jetson 실행 폴더로 이동합니다.

```bash
cd /home/jetson/Downloads/final_runtime
```

3. 노트북 IP를 넣어서 최종 코드를 실행합니다.

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 jetson_main.py --serial-port /dev/ttyUSB0 --video-udp-host <NOTEBOOK_IP>
```

예시:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 jetson_main.py --serial-port /dev/ttyUSB0 --video-udp-host 10.82.166.123
```

`<NOTEBOOK_IP>`는 Jetson 영상을 받을 노트북의 IP 주소입니다.

## 포트 연결

- 노트북 -> Jetson 페달/핸들 UDP: `5005`
- Jetson -> 노트북 RTP/JPEG 영상 UDP: `5000`
- Jetson -> 노트북 보정 페달 telemetry UDP: `5006`

현재 노트북 최종 코드는 아래 형식으로 Jetson에 입력을 보냅니다.

```text
time_sec,pedal_percent,steering,brake,mode,emergency
```

## 정리 요약

- `jetson_main.py`가 최종 Jetson 실행 코드입니다.
- `11_jetson_realtime_control_with_video.py`를 기반으로 만들었고, `src/model_runtime.py` 의존 코드를 내부에 통합했습니다.
- 따라서 다른 Python 코드 파일을 import하지 않고 `jetson_main.py` 하나로 실행됩니다.
- 단, 모델/설정/이미지 같은 실행 자료는 필요합니다.

필수 실행 자료:

- `config/runtime_config.json`
- `weights/best.pt`
- `samples/`
- `mode_jpg/`
- `pedal_baseline_state.json`
- `pedal_baseline_state_initial.json`
- `pedal_baseline_state_warmup.json`

## 설치 항목

Jetson 환경에서 필요한 주요 패키지:

```bash
python3 -m pip install --user -r requirements-jetson-jp62.txt
python3 -m pip install --user pillow opencv-python matplotlib numpy pyserial
```

JetPack/PyTorch는 장비 환경에 맞는 aarch64 CUDA wheel이 필요합니다. 현재 번들 기준은 `torch==2.8.0`, `torchvision==0.23.0`입니다.

## 개발 흐름

- `01_send_pedal_udp.py`

  CSV 페달 입력을 UDP로 보내는 초기 송신 테스트 코드입니다.

- `02_pedal_analysis.py`

  페달 이벤트를 분석하고 Gaussian baseline 평균/분산을 학습하는 핵심 실험 코드입니다.

- `03_simulate_repeated_runs.py`

  같은 입력을 반복 실행하면서 baseline이 안정적으로 유지되는지 확인한 시뮬레이션 코드입니다.

- `04_animate_pedal_correction.py`

  raw 페달과 보정 페달을 시간 흐름에 따라 시각화한 애니메이션 코드입니다.

- `05_compare_k_curves.py`

  보정 강도 함수 `k = r^2`와 `k = r^3`를 비교한 실험 코드입니다.

- `06_plot_pedal_correction.py`

  정지 이미지/CSV 기반으로 보정 결과를 비교하던 초기 시각화 코드입니다.

- `07_motorControl.py`

  CSV 기반으로 Arduino serial 모터 명령을 보내던 초기 제어 테스트 코드입니다.

- `08_jetson_realtime_control.py`

  과거 실제 실행에 사용했던 Jetson 실시간 제어 코드입니다. UDP 페달 입력을 받아 보정하고 Arduino로 전송하지만, 노트북 영상 스트리밍 기능은 없습니다.

- `09_usb_camera_viewer.py`

  USB 카메라 입력과 Jetson 화면/UDP 스트림을 확인하던 테스트 코드입니다.

- `10_usb_camera_stream_overlay.py`

  USB 카메라 영상 위에 상태 overlay를 합성해 스트리밍하던 중간 코드입니다.

- `11_jetson_realtime_control_with_video.py`

  실시간 페달 보정, Arduino 제어, 환경 추론, USB 카메라 영상 스트리밍, telemetry 송신을 통합한 코드입니다.

- `12_usb_camera_live_inference.py`

  USB 카메라 프레임에 환경 분류 모델을 직접 적용해 live inference를 확인한 테스트 코드입니다.

- `jetson_main.py`

  최종 실행 코드입니다. `11_jetson_realtime_control_with_video.py`의 기능을 유지하면서 모델 런타임 코드 의존성을 내부 통합했습니다.

## 전문가 초기분포 A 백업

아래 값은 전문가 초기분포 A 백업용이므로 삭제하면 안 됩니다.

```json
{
  "sample_count": 32.0,
  "mean": 54.843750000003936,
  "m2": 6141.796875004976,
  "variance": 198.1224798388702,
  "sigma": 14.075598738201874,
  "learned_event_count": 33
}
```
