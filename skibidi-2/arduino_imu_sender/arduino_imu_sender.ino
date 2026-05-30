/*
  arduino_imu_sender.ino
  ----------------------
  Streams IMU data over Serial for real-time gesture classification.
  Works with realtime.py (inference) and record_imu.py (data collection).

  Output: CSV lines at ~100 Hz
      time_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z

  Sensor: Adafruit LSM6DS3TR-C
  Baud:   115200
*/

#include <Adafruit_LSM6DS3TRC.h>

Adafruit_LSM6DS3TRC lsm6ds3trc;

void setup() {
  Serial.begin(115200);
  unsigned long t = millis();
  while (!Serial && millis() - t < 3000) delay(10);

  if (!lsm6ds3trc.begin_I2C()) {
    Serial.println("Failed to find LSM6DS3TR-C");
    while (1) delay(10);
  }

  lsm6ds3trc.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
  lsm6ds3trc.setGyroRange(LSM6DS_GYRO_RANGE_500_DPS);
  lsm6ds3trc.setAccelDataRate(LSM6DS_RATE_104_HZ);
  lsm6ds3trc.setGyroDataRate(LSM6DS_RATE_104_HZ);

  Serial.println("time_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z");
}

void loop() {
  sensors_event_t accel, gyro, temp;
  lsm6ds3trc.getEvent(&accel, &gyro, &temp);

  Serial.print(millis());
  Serial.print(",");

  Serial.print(accel.acceleration.x);
  Serial.print(",");
  Serial.print(accel.acceleration.y);
  Serial.print(",");
  Serial.print(accel.acceleration.z);
  Serial.print(",");

  Serial.print(gyro.gyro.x);
  Serial.print(",");
  Serial.print(gyro.gyro.y);
  Serial.print(",");
  Serial.println(gyro.gyro.z);

  delay(10);
}
