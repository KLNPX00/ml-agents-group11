import subprocess, signal, re, os
import time, math

THRESHOLD = 4.3
TIMESTAMP = math.floor(time.time())

pattern = re.compile(r"Mean Reward: (\d+\.\d+)")

WIN_CMD = ["mlagents-learn","config/ppo/PushBlock.yaml","--env=./build/UnityEnvironment.exe","--run-id=threshold_{THRESHOLD}_t_{TIMESTAMP}","--no-graphics",]
UNIX_CMD = ["mlagents-learn","config/ppo/PushBlock.yaml","--env=Project/Builds.app","--run-id=threshold_{THRESHOLD}_t_{TIMESTAMP}","--no-graphics","--base-port=6008"]

# recursive method
def run_training(threshold):
    # checks OS env
    now = math.floor(time.time())
    if threshold >= 4.95:
        threshold = 4.3

    if os.name == "nt":
        cmd = [arg.replace("{THRESHOLD}_t_{TIMESTAMP}", f"{threshold:.2f}_t_{now}") for arg in WIN_CMD]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        cmd = [arg.replace("{THRESHOLD}_t_{TIMESTAMP}", f"{threshold:.2f}_t_{now}") for arg in UNIX_CMD]
        creationflags = 0

    print(f"Starting training with threshold {threshold:.2f}, timestamp: {now}")
    p = subprocess.Popen( cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,bufsize=1, creationflags=creationflags )

    # same code as before just with OS validation
    try:
        for line in p.stdout:
            print(line, end="")
            match = pattern.search(line)
            if match and float(match.group(1)) >= threshold:
                print(f"Mean Reward has reached {threshold:.2f}, stopping training. timestamp: {now}.")

                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    p.terminate()

                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()

                print(f"📊 Exporting metrics for threshold {threshold:.2f}...")
                subprocess.run(
                    ["python", "export_tb_to_excel.py", f"{threshold:.2f}", f"{now}"],
                    check=False
                )
                print(f"✔ Excel file saved for threshold {threshold:.2f}")

                run_training(threshold + 0.05)

                return  # makes sure we exist the curr run to avoid crazy nesting

    # same safety net as before
    except KeyboardInterrupt:
        if os.name == "nt":
            p.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            p.terminate()
        p.wait()

run_training(THRESHOLD)
