import subprocess, signal, re, os
import time, math

THRESHOLD = 4.30
TIMESTAMP = math.floor(time.time())
STARTPORT=6006
pattern = re.compile(r"Mean Reward: (\d+\.\d+)")

WIN_CMD = ["mlagents-learn","config/ppo/PushBlock.yaml","--env=./build/UnityEnvironment.exe","--run-id=threshold_{THRESHOLD}_t_{TIMESTAMP}","--no-graphics","--base-port={STARTPORT}"]
UNIX_CMD = ["mlagents-learn","config/ppo/PushBlock.yaml","--env=Project/Builds.app","--run-id=threshold_{THRESHOLD}_t_{TIMESTAMP}","--no-graphics","--base-port={STARTPORT}"]

# recursive methods
def run_training(threshold,port):
    # checks OS env
    now = math.floor(time.time())
    if threshold >= 4.95:
        threshold = 4.3
     
    if port == 6006:
        port = 6007
    else: port=6006

    if os.name == "nt":
        cmd = [arg.format(THRESHOLD=f"{threshold:.2f}",TIMESTAMP=now,STARTPORT=port) for arg in WIN_CMD]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        cmd = [arg.format(THRESHOLD=f"{threshold:.2f}",TIMESTAMP=now,STARTPORT=port) for arg in UNIX_CMD]
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
                time.sleep(5)
                run_training(threshold + 0.05,port)

                return  # makes sure we exist the curr run to avoid crazy nesting

    # same safety net as before
    except KeyboardInterrupt:
        if os.name == "nt":
            p.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            p.terminate()
        p.wait()

run_training(THRESHOLD,STARTPORT)
