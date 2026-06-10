const int ENA = 3;
const int IN1 = 4;
const int IN2 = 5;
const int IN3 = 13;
const int IN4 = 12;
const int ENB = 11;

String rxBuffer = "";

int targetSpeed = 0;     // Python sends target PWM: 100~255, or STOP
float currentSpeed = 0;  // Actual applied speed

unsigned long lastUpdateMs = 0;
unsigned long boostStartMs = 0;

const float DECEL_RATE = 40.0;   // PWM per second
const int MIN_DRIVE_PWM = 150;
const int START_BOOST_PWM = 255;
const unsigned long START_BOOST_MS = 500;

bool startBoostActive = false;

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);

  Serial.begin(9600);
  stopCar();
  lastUpdateMs = millis();
}

void loop() {
  readSerialInput();
  updateSpeed();
  applyMotor();
}

void readSerialInput() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (rxBuffer.length() > 0) {
        rxBuffer.trim();

        if (rxBuffer.equalsIgnoreCase("STOP")) {
          setTargetSpeed(0);
        } else {
          int value = rxBuffer.toInt();

          if (value >= 100 && value <= 255) {
            setTargetSpeed(value);
          }
        }

        rxBuffer = "";
      }
    } else {
      rxBuffer += c;
    }
  }
}

void setTargetSpeed(int value) {
  bool startingFromStop = (targetSpeed == 0 && currentSpeed <= 0 && value > 0);

  targetSpeed = value;

  if (startingFromStop) {
    startBoostActive = true;
    boostStartMs = millis();
  }

  if (targetSpeed == 0) {
    startBoostActive = false;
  }
}

void updateSpeed() {
  unsigned long now = millis();
  float dt = (now - lastUpdateMs) / 1000.0;
  if (dt <= 0) return;
  lastUpdateMs = now;

  if (targetSpeed > currentSpeed) {
    currentSpeed = targetSpeed;
  } else if (targetSpeed < currentSpeed) {
    currentSpeed -= DECEL_RATE * dt;

    if (currentSpeed < targetSpeed) {
      currentSpeed = targetSpeed;
    }
  }

  if (currentSpeed < 100 && targetSpeed == 0) {
    currentSpeed = 0;
  }
}

void applyMotor() {
  int pwm = (int)currentSpeed;

  if (pwm <= 0) {
    stopCar();
    return;
  }

  int drivePwm = pwm;

  if (drivePwm > 0 && drivePwm < MIN_DRIVE_PWM) {
    drivePwm = MIN_DRIVE_PWM;
  }

  if (startBoostActive) {
    if (millis() - boostStartMs < START_BOOST_MS) {
      drivePwm = START_BOOST_PWM;
    } else {
      startBoostActive = false;
    }
  }

  setForwardDirection();

  analogWrite(ENA, drivePwm);
  analogWrite(ENB, drivePwm);
}

void setForwardDirection() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void stopCar() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
