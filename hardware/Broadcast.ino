#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// WiFi Configuration
const char* ssid = ""; //removed before git upload
const char* password = ""; //removed before git upload

// sending to .255 allows both Mac and Pi to hear the message
IPAddress broadcastIP(192, 168, 1, 255); 
const int udpPort = 5005;

Adafruit_ADS1115 ads;
WiFiUDP udp;

// Buffer for batching samples
#define BUFFER_SIZE 50
struct Sample {
    int16_t value;
} __attribute__((packed));

Sample sampleBuffer[BUFFER_SIZE];
uint16_t bufferIndex = 0;

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    // Initialize I2C and ADS1115
    Wire.begin();
    Wire.setClock(400000);
    
    if (!ads.begin(0x48)) {
        Serial.println("ADS1115 not found");
        while (1);
    }
    
    ads.setGain(GAIN_ONE);
    ads.setDataRate(RATE_ADS1115_860SPS);
    
    // Connect to WiFi
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        attempts++;
    }
    
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi connection failed");
        while(1);
    }
    
    Serial.println("Connected");
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Broadcasting to: ");
    Serial.print(broadcastIP);
    Serial.print(":");
    Serial.println(udpPort);
    
    udp.begin(udpPort);
}

void loop() {
    int16_t adcValue = ads.readADC_SingleEnded(0);
    
    sampleBuffer[bufferIndex].value = adcValue;
    bufferIndex++;
    
    if (bufferIndex >= BUFFER_SIZE) {
        udp.beginPacket(broadcastIP, udpPort);
        udp.write((uint8_t*)sampleBuffer, sizeof(sampleBuffer));
        udp.endPacket();
        bufferIndex = 0;
    }
}