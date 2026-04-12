/*
 * SurfaceIdiot - Data Glove Firmware
 * ESP32-WROOM-32
 *
 * Hardware:
 *   - 5x Flex sensors (GPIO34,35,32,33,25) via 10kΩ voltage divider to 3.3V
 *   - 2x MPU-6050 IMU via I2C (SDA=GPIO21, SCL=GPIO22)
 *       IMU #1 hand back : addr 0x68 (AD0 → GND)
 *       IMU #2 wrist      : addr 0x69 (AD0 → 3.3V)
 *
 * Output: JSON on Serial @ 115200 baud, ~30 Hz
 *   {"ts":12345,"fingers":{"thumb":0.42,...},"hand_imu":{...},"wrist_imu":{...}}
 *
 * Calibration values (flex_min / flex_max) are stored in NVS and updated
 * over Serial by sending the commands:
 *   "CAL_STRAIGHT\n"  — record straight baseline
 *   "CAL_FIST\n"      — record fist baseline and save
 */

#include <Wire.h>
#include <MPU6050.h>
#include <ArduinoJson.h>
#include <Preferences.h>

// ─── Pin definitions ────────────────────────────────────────────────────────
const int FLEX_PINS[]        = {34, 35, 32, 33, 25};
const int NUM_FLEX           = 5;
const char* FINGER_NAMES[]   = {"thumb","index","middle","ring","pinky"};

// I2C
const int SDA_PIN = 21;
const int SCL_PIN = 22;

// ─── Calibration (NVS-backed) ───────────────────────────────────────────────
Preferences prefs;
float flex_min[5] = {1500, 1500, 1500, 1500, 1500}; // raw ADC straight
float flex_max[5] = {3000, 3000, 3000, 3000, 3000}; // raw ADC full fist

// ─── IMU objects ─────────────────────────────────────────────────────────────
MPU6050 imu_hand(0x68);
MPU6050 imu_wrist(0x69);

// ─── Helpers ─────────────────────────────────────────────────────────────────
void loadCalibration() {
    prefs.begin("glove", true); // read-only
    for (int i = 0; i < NUM_FLEX; i++) {
        char key_min[12], key_max[12];
        snprintf(key_min, sizeof(key_min), "min_%d", i);
        snprintf(key_max, sizeof(key_max), "max_%d", i);
        flex_min[i] = prefs.getFloat(key_min, flex_min[i]);
        flex_max[i] = prefs.getFloat(key_max, flex_max[i]);
    }
    prefs.end();
}

void saveCalibration() {
    prefs.begin("glove", false); // read-write
    for (int i = 0; i < NUM_FLEX; i++) {
        char key_min[12], key_max[12];
        snprintf(key_min, sizeof(key_min), "min_%d", i);
        snprintf(key_max, sizeof(key_max), "max_%d", i);
        prefs.putFloat(key_min, flex_min[i]);
        prefs.putFloat(key_max, flex_max[i]);
    }
    prefs.end();
}

float readNormalized(int pin_idx) {
    int raw = analogRead(FLEX_PINS[pin_idx]);
    float span = flex_max[pin_idx] - flex_min[pin_idx];
    if (span < 1.0f) span = 1.0f; // avoid div-by-zero
    float v = (raw - flex_min[pin_idx]) / span;
    return constrain(v, 0.0f, 1.0f);
}

// Average N ADC readings to reduce noise
int stableADC(int pin, int samples = 16) {
    long sum = 0;
    for (int i = 0; i < samples; i++) {
        sum += analogRead(pin);
        delayMicroseconds(200);
    }
    return (int)(sum / samples);
}

// ─── Setup ───────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Wire.begin(SDA_PIN, SCL_PIN);

    // Init IMUs
    imu_hand.initialize();
    imu_hand.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
    imu_hand.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);
    imu_hand.setDLPFMode(MPU6050_DLPF_BW_42);

    imu_wrist.initialize();
    imu_wrist.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
    imu_wrist.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);
    imu_wrist.setDLPFMode(MPU6050_DLPF_BW_42);

    // Load saved calibration
    loadCalibration();

    // Confirm connection
    if (!imu_hand.testConnection())  Serial.println("# ERR: hand IMU not found");
    if (!imu_wrist.testConnection()) Serial.println("# ERR: wrist IMU not found");

    Serial.println("# SurfaceIdiot Glove Firmware ready");
}

// ─── Calibration state machine ───────────────────────────────────────────────
enum CalState { IDLE, COLLECTING_STRAIGHT, COLLECTING_FIST };
CalState cal_state = IDLE;
float cal_acc[5]   = {};
int   cal_count    = 0;
const int CAL_SAMPLES = 150; // 5s @ 30Hz

void handleCalibration() {
    if (cal_state == IDLE) return;

    int raw[5];
    for (int i = 0; i < NUM_FLEX; i++) raw[i] = stableADC(FLEX_PINS[i], 8);

    for (int i = 0; i < NUM_FLEX; i++) cal_acc[i] += raw[i];
    cal_count++;

    if (cal_count >= CAL_SAMPLES) {
        if (cal_state == COLLECTING_STRAIGHT) {
            for (int i = 0; i < NUM_FLEX; i++) flex_min[i] = cal_acc[i] / CAL_SAMPLES;
            Serial.println("# CAL_STRAIGHT_DONE");
        } else {
            for (int i = 0; i < NUM_FLEX; i++) flex_max[i] = cal_acc[i] / CAL_SAMPLES;
            saveCalibration();
            Serial.println("# CAL_FIST_DONE - calibration saved");
        }
        cal_state = IDLE;
        cal_count = 0;
        memset(cal_acc, 0, sizeof(cal_acc));
    }
}

void checkSerialCommands() {
    if (!Serial.available()) return;
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "CAL_STRAIGHT") {
        cal_state = COLLECTING_STRAIGHT;
        cal_count = 0;
        memset(cal_acc, 0, sizeof(cal_acc));
        Serial.println("# Collecting straight baseline for 5s...");
    } else if (cmd == "CAL_FIST") {
        cal_state = COLLECTING_FIST;
        cal_count = 0;
        memset(cal_acc, 0, sizeof(cal_acc));
        Serial.println("# Collecting fist baseline for 5s...");
    } else if (cmd == "STATUS") {
        Serial.printf("# min: %.0f %.0f %.0f %.0f %.0f\n",
            flex_min[0],flex_min[1],flex_min[2],flex_min[3],flex_min[4]);
        Serial.printf("# max: %.0f %.0f %.0f %.0f %.0f\n",
            flex_max[0],flex_max[1],flex_max[2],flex_max[3],flex_max[4]);
    }
}

// ─── Main loop ───────────────────────────────────────────────────────────────
void loop() {
    unsigned long t0 = millis();

    checkSerialCommands();
    handleCalibration();

    // Skip streaming during calibration collection
    if (cal_state != IDLE) {
        delay(33);
        return;
    }

    // ── Read sensors ──
    int16_t ax, ay, az, gx, gy, gz;
    imu_hand.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    int16_t ax2, ay2, az2, gx2, gy2, gz2;
    imu_wrist.getMotion6(&ax2, &ay2, &az2, &gx2, &gy2, &gz2);

    // ── Build JSON ──
    StaticJsonDocument<512> doc;
    doc["ts"] = millis();

    JsonObject fingers = doc.createNestedObject("fingers");
    for (int i = 0; i < NUM_FLEX; i++) {
        fingers[FINGER_NAMES[i]] = readNormalized(i);
    }

    // accel in g (±4g range → /8192), gyro in deg/s (±500 → /65.5)
    JsonObject hi = doc.createNestedObject("hand_imu");
    hi["ax"] = ax  / 8192.0f;   hi["ay"] = ay  / 8192.0f;   hi["az"] = az  / 8192.0f;
    hi["gx"] = gx  / 65.5f;     hi["gy"] = gy  / 65.5f;     hi["gz"] = gz  / 65.5f;

    JsonObject wi = doc.createNestedObject("wrist_imu");
    wi["ax"] = ax2 / 8192.0f;   wi["ay"] = ay2 / 8192.0f;   wi["az"] = az2 / 8192.0f;
    wi["gx"] = gx2 / 65.5f;     wi["gy"] = gy2 / 65.5f;     wi["gz"] = gz2 / 65.5f;

    serializeJson(doc, Serial);
    Serial.println();

    // Target 30 Hz
    long elapsed = millis() - t0;
    if (elapsed < 33) delay(33 - elapsed);
}
