최종 업로드 코드: `02_arduino_vehicle_final.ino`

# Arduino Vehicle Code

## 최종 사용 방법

Arduino IDE에서 `02_arduino_vehicle_final.ino`를 열고 차량 아두이노 보드에 업로드합니다.

업로드 후 노트북/Jetson 쪽에서 시리얼로 보내는 PWM 명령을 받아 차량 모터를 제어합니다.

## 정리 요약

- `02_arduino_vehicle_final.ino`가 최종 차량 포팅 코드입니다.
- 기존 `아두이노 포팅코드.txt`와 최종 `.ino` 내용이 동일한 것을 확인한 뒤 txt 파일은 제거했습니다.
- 디버깅용 중간 코드는 `01_`로 남기고, 최종 코드는 `02_`로 정리했습니다.
- 현재 이 폴더는 `.ino` 코드 파일과 이 README만 남겨둔 상태입니다.

## 개발 흐름

- `01_motor_full_pwm_debug.ino`

  모터 PWM 출력과 기본 구동을 확인하던 중간 디버깅 코드입니다. LED 상태값 처리 등 실험 기능이 포함되어 있습니다.

- `02_arduino_vehicle_final.ino`

  최종 아두이노 차량 포팅 코드입니다. 시리얼 입력으로 `STOP` 또는 `100~255` PWM 값을 받아 목표 속도를 설정하고, 감속률과 시작 부스트를 적용해 차량 모터를 제어합니다.

## 별도 설치 항목

추가 라이브러리는 사용하지 않습니다. Arduino 기본 함수(`Serial`, `pinMode`, `digitalWrite`, `analogWrite`, `millis`)만 사용합니다.
