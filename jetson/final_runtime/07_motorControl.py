import csv
import time
import serial

# ===== 사용자 설정 =====
PORT = "COM5"   # 본인 아두이노 포트로 수정
BAUDRATE = 9600
CSV_PATH = "pedal_input_sequence_deadzone_pull_rounded_1dp_120s.csv"

# CSV 컬럼 후보
TIME_COL_CANDIDATES = ["time_sec", "time_s", "time", "시간s", "시간", "t"]
VALUE_COL_CANDIDATES = ["pedal_percent", "mapped_value", "input", "입력", "value", "speed"]

def find_column(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
            return c
    return None

def pedal_percent_to_pwm(pedal_percent):
    """
    pedal_percent: 0~100
    pwm: 100~255
    """
    if pedal_percent < 0:
        pedal_percent = 0
    if pedal_percent > 100:
        pedal_percent = 100

    pwm = 100 + (pedal_percent / 100.0) * (255 - 100)
    return round(pwm)

def load_sequence(csv_path):
    sequence = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        time_col = find_column(reader.fieldnames, TIME_COL_CANDIDATES)
        value_col = find_column(reader.fieldnames, VALUE_COL_CANDIDATES)

        if time_col is None or value_col is None:
            raise ValueError(
                f"CSV 컬럼을 찾지 못했습니다.\n"
                f"현재 컬럼: {reader.fieldnames}\n"
                f"time 컬럼 후보: {TIME_COL_CANDIDATES}\n"
                f"value 컬럼 후보: {VALUE_COL_CANDIDATES}"
            )

        for row in reader:
            t = float(row[time_col])
            pedal_percent = float(row[value_col])
            sequence.append((t, pedal_percent))

    return sequence

def send_line(ser, text):
    ser.write((text + "\n").encode("utf-8"))

def main():
    sequence = load_sequence(CSV_PATH)
    print(f"총 {len(sequence)}개 포인트 로드 완료")

    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2.0)  # 아두이노 리셋 대기

    print("아두이노 연결 완료")
    print("시나리오 시작")

    start_time = time.perf_counter()
    idx = 0

    try:
        while idx < len(sequence):
            elapsed = time.perf_counter() - start_time
            target_t, pedal_percent = sequence[idx]

            if elapsed >= target_t:
                # 0%면 정지
                if pedal_percent <= 0.1:
                    send_line(ser, "STOP")
                    print(f"[{elapsed:6.2f}s] pedal={pedal_percent:5.1f}% -> STOP")
                else:
                    pwm = pedal_percent_to_pwm(pedal_percent)
                    send_line(ser, str(pwm))
                    print(f"[{elapsed:6.2f}s] pedal={pedal_percent:5.1f}% -> send {pwm}")

                idx += 1
            else:
                time.sleep(0.002)

        send_line(ser, "STOP")
        print("시나리오 종료, STOP 전송")

    except KeyboardInterrupt:
        send_line(ser, "STOP")
        print("사용자 중단, STOP 전송")

    finally:
        time.sleep(0.2)
        ser.close()
        print("포트 종료")

if __name__ == "__main__":
    main()