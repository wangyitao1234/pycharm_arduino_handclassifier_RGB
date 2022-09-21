#include <cvzone.h>

SerialData serialData(3,1);
int ValsRec[3];

int red = 8;
int blue = 9;
int green = 10;

void setup() {
  // put your setup code here, to run once:
  serialData.begin();
  pinMode(red, OUTPUT);
  pinMode(blue, OUTPUT);
  pinMode(green, OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  serialData.Get(ValsRec); 
  digitalWrite(red, ValsRec[0]);
  digitalWrite(blue, ValsRec[2]);
  digitalWrite(green, ValsRec[1]);
  delay(1);
  
}
