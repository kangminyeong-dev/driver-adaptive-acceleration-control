import ctypes
import time
from ctypes import wintypes


JOY_RETURNX = 0x00000001
JOY_RETURNY = 0x00000002
JOY_RETURNZ = 0x00000004
JOY_RETURNR = 0x00000008
JOY_RETURNU = 0x00000010
JOY_RETURNV = 0x00000020
JOY_RETURNPOV = 0x00000040
JOY_RETURNBUTTONS = 0x00000080
JOY_RETURNALL = (
    JOY_RETURNX
    | JOY_RETURNY
    | JOY_RETURNZ
    | JOY_RETURNR
    | JOY_RETURNU
    | JOY_RETURNV
    | JOY_RETURNPOV
    | JOY_RETURNBUTTONS
)

JOYERR_NOERROR = 0
JOYERR_UNPLUGGED = 167


class JOYINFOEX(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("dwXpos", wintypes.DWORD),
        ("dwYpos", wintypes.DWORD),
        ("dwZpos", wintypes.DWORD),
        ("dwRpos", wintypes.DWORD),
        ("dwUpos", wintypes.DWORD),
        ("dwVpos", wintypes.DWORD),
        ("dwButtons", wintypes.DWORD),
        ("dwButtonNumber", wintypes.DWORD),
        ("dwPOV", wintypes.DWORD),
        ("dwReserved1", wintypes.DWORD),
        ("dwReserved2", wintypes.DWORD),
    ]


winmm = ctypes.WinDLL("winmm")
joyGetNumDevs = winmm.joyGetNumDevs
joyGetNumDevs.restype = wintypes.UINT

joyGetPosEx = winmm.joyGetPosEx
joyGetPosEx.argtypes = [wintypes.UINT, ctypes.POINTER(JOYINFOEX)]
joyGetPosEx.restype = wintypes.UINT


def read_device(device_id):
    info = JOYINFOEX()
    info.dwSize = ctypes.sizeof(JOYINFOEX)
    info.dwFlags = JOY_RETURNALL
    result = joyGetPosEx(device_id, ctypes.byref(info))
    if result != JOYERR_NOERROR:
        return None, result
    return info, result


def find_connected_devices():
    connected = []
    for device_id in range(joyGetNumDevs()):
        info, result = read_device(device_id)
        if result == JOYERR_NOERROR:
            connected.append((device_id, info))
    return connected


def pov_to_text(pov):
    if pov == 0xFFFF:
        return "center"
    degrees = pov / 100
    directions = {
        0: "up",
        45: "up-right",
        90: "right",
        135: "down-right",
        180: "down",
        225: "down-left",
        270: "left",
        315: "up-left",
    }
    return directions.get(int(degrees), f"{degrees:.2f} deg")


def pressed_button_numbers(button_bits):
    return [index + 1 for index in range(32) if button_bits & (1 << index)]


def axis_values(info):
    return {
        "X": info.dwXpos,
        "Y": info.dwYpos,
        "Z": info.dwZpos,
        "R": info.dwRpos,
        "U": info.dwUpos,
        "V": info.dwVpos,
    }


def print_snapshot(device_id, info):
    axes = axis_values(info)
    axes_text = " ".join(f"{name}={value:5d}" for name, value in axes.items())
    buttons = pressed_button_numbers(info.dwButtons)
    print(f"[device {device_id}] {axes_text} POV={pov_to_text(info.dwPOV)} buttons={buttons}")


def main():
    devices = find_connected_devices()
    if not devices:
        print("연결된 조이스틱/휠 장치를 찾지 못했습니다.")
        print("Windows 게임 컨트롤러 설정에서 장치가 보이는지 먼저 확인해보세요.")
        return

    print("감지된 입력 장치:")
    for device_id, info in devices:
        print_snapshot(device_id, info)

    print()
    print("버튼을 누르거나 휠/페달을 움직이면 변화가 표시됩니다. 종료: Ctrl+C")
    print("버튼 번호는 Windows joystick API 기준으로 1번부터 표시합니다.")
    print()

    previous = {device_id: info for device_id, info in devices}
    axis_threshold = 400

    try:
        while True:
            for device_id in list(previous):
                info, result = read_device(device_id)
                if result == JOYERR_UNPLUGGED:
                    print(f"[device {device_id}] 연결 해제됨")
                    del previous[device_id]
                    continue
                if result != JOYERR_NOERROR:
                    continue

                old = previous[device_id]
                old_buttons = old.dwButtons
                new_buttons = info.dwButtons

                changed_buttons = old_buttons ^ new_buttons
                for index in range(32):
                    bit = 1 << index
                    if not changed_buttons & bit:
                        continue
                    state = "DOWN" if new_buttons & bit else "UP"
                    print(f"[device {device_id}] BUTTON {index + 1:02d} {state}")

                if old.dwPOV != info.dwPOV:
                    print(f"[device {device_id}] POV {pov_to_text(old.dwPOV)} -> {pov_to_text(info.dwPOV)}")

                old_axes = axis_values(old)
                new_axes = axis_values(info)
                changed_axes = [
                    f"{name}:{old_axes[name]}->{new_axes[name]}"
                    for name in new_axes
                    if abs(int(new_axes[name]) - int(old_axes[name])) >= axis_threshold
                ]
                if changed_axes:
                    print(f"[device {device_id}] AXIS " + " ".join(changed_axes))

                previous[device_id] = info

            if not previous:
                time.sleep(1.0)
                for device_id, info in find_connected_devices():
                    if device_id not in previous:
                        previous[device_id] = info
                        print(f"[device {device_id}] 다시 연결됨")
                        print_snapshot(device_id, info)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
