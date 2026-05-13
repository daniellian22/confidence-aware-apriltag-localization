/*
 * Robust Dual Linear Actuator Control (newline-delimited protocol)
 *
 * Protocol (compatible with your current MATLAB tcp_handshake):
 *   1) Host sends one number terminated by '\n':
 *        - X command: 0..900        (mm; can be fractional e.g., 7.6)
 *        - Y command: 1000..1500    (1000 + y_mm)
 *   2) Arduino replies: (value + 123)
 *   3) Host confirms by sending back exactly that echoed number (value + 123) within timeout
 *   4) Arduino moves the requested axis and blinks LED once on completion.
 *
 * Motion model (kept consistent with your current code):
 *   - Internal units: 0.1 mm (integer) to avoid float drift
 *   - For each 0.1 mm: generate (STEPS_PER_REV/100) pulses
 */

 #include <Arduino.h>
 #include <math.h>
 #include <stdlib.h>
 
 // ---------------- Pins ----------------
 static const uint8_t LED_PIN = 13;
 
 // Actuator 1 (X)
 static const uint8_t PUL_X = 7;
 static const uint8_t DIR_X = 6;
 static const uint8_t ENA_X = 5;
 
 // Actuator 2 (Y)
 static const uint8_t PUL_Y = 12;
 static const uint8_t DIR_Y = 10;
 static const uint8_t ENA_Y = 3;
 
 // ---------------- Settings ----------------
 static const long STEPS_PER_REV = 6400;
 static const long STEPS_PER_0P1MM = (STEPS_PER_REV / 100); // same assumption as your original code
 static const unsigned int PULSE_USEC = 150;
 
 static const float X_MAX_MM = 900.0f;
 static const float Y_MAX_MM = 500.0f;
 static const float Y_OFFSET = 1000.0f;
 
 static const unsigned long CONFIRM_TIMEOUT_MS = 3000; // extended confirm window
 static const float CONFIRM_TOL = 1e-2f;
 
 // Internal state: position in 0.1mm units
 static long currX_u01 = 10; // ~1.0 mm
 static long currY_u01 = 10; // ~1.0 mm
 
 // ---------------- Helpers ----------------
 static void blink_confirm() {
   digitalWrite(LED_PIN, HIGH);
   delay(80);
   digitalWrite(LED_PIN, LOW);
   delay(80);
 }
 
 static bool parseFloatStrict(const char *s, float &out) {
   char *endp = nullptr;
   double v = strtod(s, &endp);
   if (endp == s) return false; // no parse
   out = (float)v;
   return true;
 }
 
 static long mm_to_u01(float mm) {
   // mm -> 0.1mm integer units with rounding (avoids floor bias and float drift)
   return (long)lroundf(mm * 10.0f);
 }
 
 static void pulseTrain(uint8_t pulPin, unsigned long pulses) {
   for (unsigned long i = 0; i < pulses; i++) {
     digitalWrite(pulPin, HIGH);
     delayMicroseconds(PULSE_USEC);
     digitalWrite(pulPin, LOW);
     delayMicroseconds(PULSE_USEC);
   }
 }
 
 static void moveAxis_u01(uint8_t pulPin, uint8_t dirPin, uint8_t enaPin,
                          bool dirHighForPositive, long delta_u01) {
   if (delta_u01 == 0) return;
 
   digitalWrite(enaPin, HIGH);
 
   bool positive = (delta_u01 > 0);
   bool dirLevel = positive ? dirHighForPositive : !dirHighForPositive;
   digitalWrite(dirPin, dirLevel);
 
   unsigned long steps_u01 = (unsigned long)labs(delta_u01);
   unsigned long pulses = steps_u01 * (unsigned long)STEPS_PER_0P1MM;
   pulseTrain(pulPin, pulses);
 }
 
 // X: positive corresponds to DIR_X = HIGH (as in your original move_forward)
 static void gotoX_mm(float x_mm) {
   if (x_mm < 0.0f || x_mm > X_MAX_MM) return; // reject out-of-range to catch bugs
   long dest_u01 = mm_to_u01(x_mm);
   long delta = dest_u01 - currX_u01;
   if (delta == 0) { blink_confirm(); return; }
   moveAxis_u01(PUL_X, DIR_X, ENA_X, /*dirHighForPositive=*/true, delta);
   currX_u01 = dest_u01;
   blink_confirm();
 }
 
 // Y: your original forward2() used DIR_Y = LOW for positive; backward2() used HIGH
 // => positive corresponds to DIR_Y = LOW  => dirHighForPositive = false
 static void gotoY_mm(float y_mm) {
   if (y_mm < 0.0f || y_mm > Y_MAX_MM) return; // reject out-of-range to catch bugs
   long dest_u01 = mm_to_u01(y_mm);
   long delta = dest_u01 - currY_u01;
   if (delta == 0) { blink_confirm(); return; }
   moveAxis_u01(PUL_Y, DIR_Y, ENA_Y, /*dirHighForPositive=*/false, delta);
   currY_u01 = dest_u01;
   blink_confirm();
 }
 
 // Blocking read of a single newline-terminated line into buf.
 // Returns true only if a full line (ending in '\n') is received before timeout.
 static bool readLineBlocking(char *buf, size_t buflen, unsigned long timeout_ms) {
   unsigned long t0 = millis();
   size_t idx = 0;
 
   while (millis() - t0 < timeout_ms) {
     while (Serial.available() > 0) {
       char c = (char)Serial.read();
       if (c == '\r') continue;
       if (c == '\n') {
         if (idx == 0) { // empty line; keep waiting (within timeout)
           continue;
         }
         buf[idx] = '\0';
         return true;
       }
       if (idx + 1 < buflen) {
         buf[idx++] = c;
       }
       // else truncate safely
     }
     delay(1);
   }
   return false;
 }
 
 // Process one command line: echo, confirm, then move.
 static void handleCommandLine(const char *line) {
   float val = 0.0f;
   if (!parseFloatStrict(line, val)) return;
 
   float echo = val + 123.0f;
   Serial.println(echo, 3);
 
   char cfm[64];
   float confirm = 0.0f;
   if (!readLineBlocking(cfm, sizeof(cfm), CONFIRM_TIMEOUT_MS)) return;
   if (!parseFloatStrict(cfm, confirm)) return;
   if (fabsf(confirm - echo) > CONFIRM_TOL) return;
 
   if (val >= Y_OFFSET) {
     gotoY_mm(val - Y_OFFSET);
   } else {
     gotoX_mm(val);
   }
 }
 
 // ---------------- Main ----------------
 void setup() {
   pinMode(LED_PIN, OUTPUT);
 
   Serial.begin(115200);
 
   pinMode(PUL_X, OUTPUT);
   pinMode(DIR_X, OUTPUT);
   pinMode(ENA_X, OUTPUT);
 
   pinMode(PUL_Y, OUTPUT);
   pinMode(DIR_Y, OUTPUT);
   pinMode(ENA_Y, OUTPUT);
 
   digitalWrite(ENA_X, HIGH);
   digitalWrite(ENA_Y, HIGH);
 }
 
 void loop() {
   // Non-blocking, stateful line accumulation for commands:
   static char rx[64];
   static size_t rxLen = 0;
 
   while (Serial.available() > 0) {
     char c = (char)Serial.read();
     if (c == '\r') continue;
 
     if (c == '\n') {
       if (rxLen == 0) continue; // ignore empty lines
       rx[rxLen] = '\0';
       handleCommandLine(rx);
       rxLen = 0; // reset for next command
     } else {
       if (rxLen + 1 < sizeof(rx)) {
         rx[rxLen++] = c;
       }
       // else: overflow -> truncate; you can also reset rxLen=0 to hard-fail
     }
   }
 }