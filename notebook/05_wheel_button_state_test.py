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

STEERING_CENTER = 32767.5
STEERING_DEADZONE = 1500
STEERING_MIN = 0
STEERING_MAX = 65535

PEDAL_CENTER = 32767.5
PEDAL_DEADZONE = 1200
BRAKE_ON_THRESHOLD = 40000
BRAKE_OFF_THRESHOLD = 37000

MODE_BUTTON = 10
EMERGENCY_BUTTON = 9


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


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def read_device(device_id):
    info = JOYINFOEX()
    info.dwSize = ctypes.sizeof(JOYINFOEX)
    info.dwFlags = JOY_RETURNALL
    result = joyGetPosEx(device_id, ctypes.byref(info))
    if result != JOYERR_NOERROR:
        return None
    return info


def find_first_device():
    for device_id in range(joyGetNumDevs()):
        info = read_device(device_id)
        if info is not None:
            return device_id, info
    return None, None


def is_button_down(button_bits, button_number):
    return bool(button_bits & (1 << (button_number - 1)))


def x_to_steering(raw_x):
    delta = raw_x - STEERING_CENTER
    if abs(delta) <= STEERING_DEADZONE:
        return 0

    if delta < 0:
        usable_range = STEERING_CENTER - STEERING_DEADZONE - STEERING_MIN
        steering = (delta + STEERING_DEADZONE) / usable_range * 100
    else:
        usable_range = STEERING_MAX - STEERING_CENTER - STEERING_DEADZONE
        steering = (delta - STEERING_DEADZONE) / usable_range * 100

    return int(round(clamp(steering, -100, 100)))


def y_to_pedal(raw_y):
    delta = PEDAL_CENTER - raw_y
    if delta <= PEDAL_DEADZONE:
        return 0

    usable_range = PEDAL_CENTER - PEDAL_DEADZONE
    pedal = (delta - PEDAL_DEADZONE) / usable_range * 100
    return int(round(clamp(pedal, 0, 100)))


def y_to_brake(raw_y, previous_brake):
    if previous_brake:
        return 0 if raw_y <= BRAKE_OFF_THRESHOLD else 1
    return 1 if raw_y >= BRAKE_ON_THRESHOLD else 0


def format_packet(t_value, pedal, steering, brake, mode, emergency):
    return f"({t_value:.2f}, {pedal}, {steering}, {brake}, {mode}, {emergency})"


def main():
    device_id, previous = find_first_device()
    if previous is None:
        print("No joystick/wheel device found.")
        return

    start_time = time.perf_counter()
    mode = 1
    emergency = 0
    brake = 0
    last_packet = None

    print(f"[device {device_id}] start")
    print("Output: (t, pedal0-100, steering-100..100, brake0/1, mode1-4, emergency0/1)")
    print(f"Button {MODE_BUTTON}: mode 1 -> 2 -> 3 -> 4")
    print(f"Button {EMERGENCY_BUTTON}: emergency 0 -> 1 -> 0")
    print("Ctrl+C to stop.")
    print()

    try:
        while True:
            current = read_device(device_id)
            if current is None:
                print(f"[device {device_id}] disconnected")
                return

            previous_buttons = previous.dwButtons
            current_buttons = current.dwButtons

            mode_pressed = (
                not is_button_down(previous_buttons, MODE_BUTTON)
                and is_button_down(current_buttons, MODE_BUTTON)
            )
            emergency_pressed = (
                not is_button_down(previous_buttons, EMERGENCY_BUTTON)
                and is_button_down(current_buttons, EMERGENCY_BUTTON)
            )

            if mode_pressed:
                mode = 1 if mode >= 4 else mode + 1
                emergency = 0

            if emergency_pressed:
                emergency = 0 if emergency else 1

            t_value = time.perf_counter() - start_time
            pedal = y_to_pedal(current.dwYpos)
            steering = x_to_steering(current.dwXpos)
            brake = y_to_brake(current.dwYpos, brake)
            packet = (pedal, steering, brake, mode, emergency)

            if packet != last_packet or mode_pressed or emergency_pressed:
                print(format_packet(t_value, pedal, steering, brake, mode, emergency))
                print(f"  raw: X={current.dwXpos} Y={current.dwYpos} buttons={current.dwButtons}")
                last_packet = packet

            previous = current
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
