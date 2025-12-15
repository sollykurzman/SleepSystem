/*
 * ESP32-S2 Mini Switch State Change Detector (UDP Version)
 * * Logic:
 * - Uses Internal Pull-up Resistor on Pin 4.
 * - Connects to Wi-Fi.
 * - Sends UDP packets to Raspberry Pi on state change.
 */

#include <WiFi.h>
#include <WiFiUdp.h>

// Wifi config
const char* ssid = ""; //removed before git upload
const char* password = ""; //removed before git upload

// Destination config
IPAddress broadcastIP(192, 168, 1, 255); 
const int udpPort = 5006;

// Pins
const int SWITCH_PIN = 16; 

// Variables
WiFiUDP udp;
int lastSteadyState = HIGH;       // Previous stable state
int lastFlickerableState = HIGH;  // Current reading
long lastDebounceTime = 0;        
const long debounceDelay = 50;
unsigned long lastSendTime = 0;
const long sendInterval = 5000;    

void setup() {
  Serial.begin(115200);
  pinMode(SWITCH_PIN, INPUT_PULLUP);

  // Connect to WiFi
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  Serial.println("Code V1");

  // Start UDP
  udp.begin(udpPort);
}

void sendUDPPacket(String message) {
  udp.beginPacket(broadcastIP, udpPort);
  udp.print(message);
  udp.endPacket();
  Serial.println("Sent UDP: " + message);
}

void loop() {
  unsigned long currentMillis = millis();
  int currentState = digitalRead(SWITCH_PIN);

  // Debounce logic
  if (currentState != lastFlickerableState) {
    lastDebounceTime = currentMillis;
    lastFlickerableState = currentState;
  }

  if ((currentMillis - lastDebounceTime) > debounceDelay) {
    if (lastSteadyState != currentState) {
      lastSteadyState = currentState;

      // State has changed, send UDP packet
      if (lastSteadyState == LOW) {
        sendUDPPacket("notInBed");
        Serial.println("notInBed");
      } else {
        sendUDPPacket("Awake");
        Serial.println("Awake");
      }
    }
  }

  if (currentMillis - lastSendTime >= sendInterval) {
    lastSendTime = currentMillis;

    if (lastSteadyState == LOW) {
        sendUDPPacket("notInBed");
        Serial.println("notInBed");
      } else {
        sendUDPPacket("Awake");
        Serial.println("Awake");
      }
  }
}