import serial
import time

PORT = "/dev/cu.usbmodem206EF1313C102"
BAUD = 115200

letter   = input("Letter label, e.g. A: ").strip().upper()
trial    = input("Trial number, e.g. 001: ").strip()
filename = f"{letter}_{trial}.csv"

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    print(f"Could not open port: {e}")
    print("Run:  fuser -k /dev/ttyACM0   then try again.")
    raise SystemExit(1)

time.sleep(2)                  # wait for Arduino reboot after DTR reset
ser.reset_input_buffer()       # discard boot noise

print(f"Port open. Bytes waiting: {ser.in_waiting}")
if ser.in_waiting == 0:
    print("WARNING: no data yet — is Serial Monitor closed?")

print(f"Recording to {filename}")
print("Press ENTER to start.")
input()

lines_written = 0
print("Recording... Press Ctrl+C to stop.")

with open(filename, "w") as f:
    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if line:
                print(line)
                f.write(line + "\n")
                f.flush()
                lines_written += 1
    except KeyboardInterrupt:
        print(f"\nStopped. {lines_written} lines written to {filename}.")

ser.close()