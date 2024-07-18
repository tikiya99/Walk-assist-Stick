#include <Arduino.h>
#include <SPI.h>
#include <MFRC522.h>
#include <ESP32Servo.h>  

#define TRIG_PIN 13
#define ECHO_PIN 14

#define LED_PIN 16

#define SS_PIN 5 //SDA 
#define RST_PIN 22 //RST
/* Pin connection for the RFID Reader
SDA - 5
SCK - 18
MOSI -23
MISO -19
RST - 22
*/

#define IN1 26
#define IN2 25
#define IN3 33
#define IN4 32

#define SERVO_PIN 12 // Define the pin for the servo motor

MFRC522 mfrc522(SS_PIN, RST_PIN);
Servo myservo; // Create a servo object

// Function prototypes
void measureDistance();
void readRFID();
void processSerialCommands();
void controlServo();

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  SPI.begin(); // Initialize SPI bus
  mfrc522.PCD_Init(); // Initialize MFRC522

  myservo.attach(SERVO_PIN); // Attach the servo to the defined pin

  Serial.println("Setup complete");
}

void loop() {
  measureDistance();
  readRFID();
  processSerialCommands();
  controlServo();
}

void measureDistance() {
  long duration, distance;
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  duration = pulseIn(ECHO_PIN, HIGH);
  distance = duration * 0.034 / 2;

  if (distance < 20) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);  
  }
}

void readRFID() {
  if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial()) {
    Serial.println("RFID detected!");
    String content = "";
    byte letter;
    for (byte i = 0; i < mfrc522.uid.size; i++) {
      content.concat(String(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " "));
      content.concat(String(mfrc522.uid.uidByte[i], HEX));
    }
    Serial.println("ID: " + content);
    mfrc522.PICC_HaltA();
  } else {
    Serial.println("RFID not detected");
  }
}

void processSerialCommands() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    Serial.println(command);
    switch (command) {
      case '1':
        digitalWrite(IN1, HIGH);
        digitalWrite(IN2, LOW);
        digitalWrite(IN3, HIGH);
        digitalWrite(IN4, LOW);
        break;
      case '2':
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, HIGH);
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, HIGH);
        break;
      default:
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, LOW);
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, LOW);
        break;
    }
  }
}

void controlServo() {
  static int pos = 0;
  static int increment = 1;
  myservo.write(pos);
  pos += increment;
  if (pos >= 180 || pos <= 0) {
    increment = -increment;
  }
  delay(15); // Adjust this delay to control the speed of the servo sweep
}
