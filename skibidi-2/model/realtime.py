"""
Manual IMU letter recognition using TFLite.

ENTER = start recording
ENTER again = stop recording and classify
"""

import argparse
import time
import threading

import numpy as np
import serial

LABELS = ["G", "I", "N", "S"]
SEQ_LEN = 300
N_FEATURES = 6

MIN_SAMPLES = 40
CONFIDENCE_MIN = 0.55


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/cu.usbmodem206EF1313C102")
    p.add_argument("--baud", default=115200, type=int)
    p.add_argument("--model", default="imu_classifier.tflite")
    return p.parse_args()


def load_tflite(path: str):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def preprocess(raw_samples: list) -> np.ndarray:
    seq = np.array(raw_samples, dtype=np.float32)

    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-8
    seq = (seq - mean) / std

    if len(seq) >= SEQ_LEN:
        seq = seq[:SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - len(seq), N_FEATURES), dtype=np.float32)
        seq = np.vstack([seq, pad])

    return seq[np.newaxis, ...]


def predict(interp, inp_details, out_details, raw_samples: list):
    tensor = preprocess(raw_samples)
    interp.set_tensor(inp_details[0]["index"], tensor)
    interp.invoke()
    probs = interp.get_tensor(out_details[0]["index"])[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx]), probs


def parse_line(line: str):
    parts = line.strip().split(",")
    if len(parts) != 7:
        return None

    try:
        return [float(x) for x in parts[1:]]
    except ValueError:
        return None


def serial_reader(ser, recording_flag, buffer, lock, stop_flag):
    while not stop_flag["stop"]:
        try:
            raw = ser.readline()

            if not raw:
                continue

            vals = parse_line(raw.decode(errors="ignore"))

            if vals is None:
                continue

            if recording_flag["recording"]:
                with lock:
                    buffer.append(vals)

        except serial.SerialException as e:
            print(f"\nSerial disconnected or busy: {e}")
            stop_flag["stop"] = True
            break

        except OSError as e:
            print(f"\nOS serial error: {e}")
            stop_flag["stop"] = True
            break


def main():
    args = parse_args()

    interp, inp_details, out_details = load_tflite(args.model)

    print(f"Model loaded: {args.model}")
    print(f"Letters: {LABELS}")

    print(f"Opening serial port: {args.port}")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

    print(f"Serial open: {args.port} @ {args.baud}")
    print("-" * 60)
    print("Manual recording mode")
    print("ENTER = start recording")
    print("ENTER again = stop recording and classify")
    print("Type q then ENTER to quit.")
    print("-" * 60)

    recording_flag = {"recording": False}
    stop_flag = {"stop": False}
    buffer = []
    lock = threading.Lock()
    transcribed = []

    reader_thread = threading.Thread(
        target=serial_reader,
        args=(ser, recording_flag, buffer, lock, stop_flag),
        daemon=True
    )
    reader_thread.start()

    try:
        while not stop_flag["stop"]:
            cmd = input("\nPress ENTER to START recording, or q to quit: ")

            if cmd.lower().strip() == "q":
                break

            with lock:
                buffer.clear()

            ser.reset_input_buffer()
            recording_flag["recording"] = True

            print("Recording... write the letter now.")

            input("Press ENTER to STOP recording: ")

            recording_flag["recording"] = False

            with lock:
                sample = list(buffer)

            n = len(sample)

            if n < MIN_SAMPLES:
                print(f"Too few samples captured: {n}. Try writing slower.")
                continue

            letter, conf, probs = predict(
                interp,
                inp_details,
                out_details,
                sample
            )

            tag = "" if conf >= CONFIDENCE_MIN else "?"
            transcribed.append(letter + tag)

            prob_str = "  ".join(
                f"{LABELS[i]}:{probs[i]:.2f}"
                for i in range(len(LABELS))
            )

            print(
                f"\nDetected: {letter}{tag}"
                f"\nConfidence: {conf:.2f}"
                f"\nSamples: {n}"
                f"\nProbs: {prob_str}"
                f"\nTranscription: {''.join(transcribed)}"
            )

    finally:
        stop_flag["stop"] = True
        recording_flag["recording"] = False
        ser.close()
        print("\nClosed serial port.")


if __name__ == "__main__":
    main()