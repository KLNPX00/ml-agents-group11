import subprocess
import signal
import re
import os
import time
import math
import sys

# ----------------------
# Same knobs as before
# ----------------------
THRESHOLD_START = 4.40
STARTPORT = 6006

THRESHOLD_STEP = 0.05
THRESHOLD_WRAP_AT = 4.95   # same idea as your: if threshold >= 4.95 -> reset
THRESHOLD_WRAP_TO = 4.30

pattern = re.compile(r"Mean Reward: (\d+\.\d+)")

WIN_CMD = [
    "mlagents-learn",
    "config/ppo/PushBlock.yaml",
    "--env=./build/UnityEnvironment.exe",
    "--run-id=threshold_{THRESHOLD}_t_{TIMESTAMP}",
    "--no-graphics",
    "--base-port={STARTPORT}",
]
UNIX_CMD = [
    "mlagents-learn",
    "config/ppo/PushBlock.yaml",
    "--env=Project/Builds.app",
    "--run-id=threshold_{THRESHOLD}_t_{TIMESTAMP}",
    "--no-graphics",
    "--base-port={STARTPORT}",
]

EXPORT_SCRIPT = "export_tb_to_excel.py"


def wrap_threshold(threshold: float) -> float:
    """
    Same functionality as your original:
    if threshold >= 4.95: threshold = 4.3
    (done BEFORE starting a run)
    """
    threshold = round(threshold, 2)
    if threshold >= THRESHOLD_WRAP_AT:
        return THRESHOLD_WRAP_TO
    return threshold


def build_cmd(threshold: float, port: int, now: int):
    """Build OS-specific command and flags (same as before)."""
    if os.name == "nt":
        cmd_template = WIN_CMD
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        cmd_template = UNIX_CMD
        creationflags = 0

    cmd = [
        arg.format(THRESHOLD=f"{threshold:.2f}", TIMESTAMP=now, STARTPORT=port)
        for arg in cmd_template
    ]
    return cmd, creationflags


def stop_process(p: subprocess.Popen):
    """Same stop logic: Windows CTRL_BREAK_EVENT else terminate, with kill fallback."""
    if os.name == "nt":
        p.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        p.terminate()

    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()


def export_metrics(threshold: float, now: int):
    """Export TensorBoard -> Excel (more robust on Windows via sys.executable)."""
    print(f"Exporting metrics for threshold {threshold:.2f}...")
    try:
        subprocess.run(
            [sys.executable, EXPORT_SCRIPT, f"{threshold:.2f}", str(now)],
            check=False,
        )
        print(f"Excel file saved for threshold {threshold:.2f}")
    except Exception as e:

        print("Export failed:", repr(e))


def main():
    threshold = round(THRESHOLD_START, 2)
    port = STARTPORT

    while True:
        now = math.floor(time.time())
        threshold = wrap_threshold(threshold)

        cmd, creationflags = build_cmd(threshold, port, now)

        print(f"Starting training with threshold {threshold:.2f}, timestamp: {now}")
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )

        reached_threshold = False

        try:
            for line in p.stdout:
                print(line, end="")

                match = pattern.search(line)
                if match and float(match.group(1)) >= threshold:
                    reached_threshold = True
                    print(f"Mean Reward has reached {threshold:.2f}, stopping training. timestamp: {now}.")

                    stop_process(p)
                    export_metrics(threshold, now)

                    time.sleep(5)

                    threshold = round(threshold + THRESHOLD_STEP, 2)
                    port += 1
                    break

            # If the process ended without hitting the threshold (your 4.95 case)
            if not reached_threshold:
                rc = p.wait()
                print(f"mlagents-learn ended (returncode={rc}) without reaching {threshold:.2f}.")


                export_metrics(threshold, now)
                time.sleep(5)

                threshold = round(threshold + THRESHOLD_STEP, 2)
                port += 1

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: stopping current training...")
            try:
                stop_process(p)
            except Exception:
                pass
            raise


if __name__ == "__main__":
    main()
