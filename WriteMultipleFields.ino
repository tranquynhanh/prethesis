#include "WiFiEsp.h"
#include "secrets.h"
#include "ThingSpeak.h"
#include "MQ135.h"
#include <dht11.h>
#include "MQ7.h"
dht11 DHT11;
#define DHT11PIN 7

char ssid[] = "QuynhNhu"; 
char pass[] = "quynhanh";
int keyIndex = 0;      
WiFiEspClient  client;
float valSensorMQ135 = 1;
float valSensorMQ7 = 1;
float valSensorUV = 1;
float valSensorPM25 = 1;
int sensorValue;
const int sensorPin0= A0;
const int sensorPin1= A1;
const int sensorPin2= A2;
const int sensorPin3= A3;
int ledPower = 11;
const int HumidityCorrection = 10;      
const int CelsiusTemperatureCorrection = -1; 
float air_quality;
float CO_value;
float volts;
float UV_index; 
int sensor_value;
double dewPoint;
float voMeasured = 0;
float calcVoltage = 0;
float dustDensity = 0;
int humidity, temperature;
float correctedRZero;
float correctedPPM;
float ppm;
float resistance;
float rzero;
#define PIN_MQ135 A2 // MQ135 Analog Input Pin
#define DHTPIN 8 // DHT Digital Input Pin
#define A_PIN 0
#define VOLTAGE 5

// init MQ7 device
MQ7 mq7(A_PIN, VOLTAGE);
MQ135 mq135_sensor(PIN_MQ135);


String field = "field1";
String field2 = "field2";
String field3 = "field3";
String field4 = "field4";
String field5 = "field5";
String field6 = "field6";
String field7 = "field7";

#ifndef HAVE_HWSERIAL1
#include "SoftwareSerial.h"
SoftwareSerial Serial1(2, 3); // RX, TX
#define ESP_BAUDRATE  19200
#else
#define ESP_BAUDRATE  115200
#endif

unsigned long myChannelNumber = 1587927;
const char * myWriteAPIKey = "C8O17P5MJ3B6E503";

// Initialize our values
String myStatus = "";

void setup() {
  //Initialize serial and wait for port to open
  Serial.begin(115200);  // Initialize serial
	mq7.calibrate();		// calculates R0
  pinMode(ledPower,OUTPUT);
  while(!Serial){
    ; // wait for serial port to connect. Needed for Leonardo native USB port only
  }
  
  // initialize serial for ESP module  
  setEspBaudRate(ESP_BAUDRATE);
  
  while (!Serial) {
    ; // wait for serial port to connect. Needed for Leonardo native USB port only
  }

  Serial.print("Searching for ESP8266..."); 
  // initialize ESP module
  WiFi.init(&Serial1);

  // check for the presence of the shield
  if (WiFi.status() == WL_NO_SHIELD) {
    Serial.println("WiFi shield not present");
    // don't continue
    while (true);
  }
  Serial.println("found it!");
   
  ThingSpeak.begin(client);  // Initialize ThingSpeak
}

void loop() {

  // Connect or reconnect to WiFi
  if(WiFi.status() != WL_CONNECTED){
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(SECRET_SSID);
    while(WiFi.status() != WL_CONNECTED){
      WiFi.begin(ssid, pass);  
      Serial.print(".");
      delay(5000);     
    } 
    Serial.println("\nConnected.");
  }
  int chk = DHT11.read(DHT11PIN);
  humidity = DHT11.humidity + HumidityCorrection;
  temperature = DHT11.temperature + CelsiusTemperatureCorrection;
  Serial.println(humidity);
  valSensorMQ135 = getSensor135Data();
  valSensorMQ7 = getCOData();
  valSensorPM25 = getPM25();
  valSensorUV = getUVIndex();
  dewPoint = getDewPoint(temperature, humidity);  

  // set the fields with the values
  ThingSpeak.setField(1, (float)valSensorMQ135);
  ThingSpeak.setField(2, (float)valSensorMQ7);
  ThingSpeak.setField(3, (float)valSensorPM25);
  ThingSpeak.setField(4, (float)valSensorUV);
  ThingSpeak.setField(5, (float)temperature);
  ThingSpeak.setField(6, (float)humidity);
  ThingSpeak.setField(7, (float)dewPoint);


  // figure out the status message
  // set the status
  ThingSpeak.setStatus(myStatus);
  
  // write to the ThingSpeak channel
  int x = ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);
  if(x == 200){
    Serial.println("Channel update successful.");
  }
  else{
    Serial.println("Problem updating channel. HTTP error code " + String(x));
  }
  
 
  delay(300000); 
}


void setEspBaudRate(unsigned long baudrate){
  long rates[6] = {115200,74880,57600,38400,19200,9600};

  Serial.print("Setting ESP8266 baudrate to ");
  Serial.print(baudrate);
  Serial.println("...");

  for(int i = 0; i < 6; i++){
    Serial1.begin(rates[i]);
    delay(100);
    Serial1.print("AT+UART_DEF=");
    Serial1.print(baudrate);
    Serial1.print(",8,1,0,0\r\n");
    delay(100);  
  }
    
  Serial1.begin(baudrate);
}

float getSensor135Data(){

  if (isnan(humidity) || isnan(temperature)) {
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }
  
  rzero = mq135_sensor.getRZero();
  correctedRZero = mq135_sensor.getCorrectedRZero(temperature, humidity);
  resistance = mq135_sensor.getResistance();
  ppm = mq135_sensor.getPPM();
  correctedPPM = mq135_sensor.getCorrectedPPM(temperature, humidity);
  Serial.println(rzero);
  return correctedPPM;
}

float getCOData() {
  CO_value = mq7.readPpm();
  Serial.println(mq7.readPpm());
  return CO_value;
}

float getUVIndex() {
  sensor_value = analogRead(sensorPin1); 
  volts = sensor_value * 5.0 / 1024.0;
  UV_index = volts * 10;
  return UV_index;
}

double getDewPoint(int celsius, int humidity)
{
  double a = 17.271;
  double b = 237.7;
  double temp = (a * celsius) / (b + celsius) + log(humidity/100);
  double Td = (b * temp) / (a - temp);
  return Td;
}

float getPM25() {
  digitalWrite(ledPower,LOW); 
  delayMicroseconds(280);  
  voMeasured = analogRead(sensorPin3); 
  delayMicroseconds(40); 
  digitalWrite(ledPower,HIGH); 
  delayMicroseconds(9680); 
  calcVoltage = voMeasured * (5.0 / 1024); 
  dustDensity = (0.172 * calcVoltage - 0.0999);
  if (dustDensity < 0)                
  {
    dustDensity = 0.00;
  }
  Serial.println(dustDensity);
  return dustDensity;
}
