import subprocess, signal, re, os

THRESHOLD = 4.5
pattern = re.compile(r"Mean Reward: (\d+\.\d+)")

WIN_CMD = ["mlagents-learn","config/ppo/PushBlock.yaml","--env=./build/UnityEnvironment.exe","--run-id=threshold{THRESHOLD}Run","--no-graphics",]
UNIX_CMD = ["mlagents-learn","config/ppo/PushBlock.yaml","--env=Project/Builds.app","--run-id=threshold{THRESHOLD}Run","--no-graphics",]

# recursive method
def run_training(threshold):
    # checks OS env
    if os.name == "nt":
        cmd = [arg.replace("{THRESHOLD}", f"{threshold:.2f}") for arg in WIN_CMD]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        cmd = [arg.replace("{THRESHOLD}", f"{threshold:.2f}") for arg in UNIX_CMD]
        creationflags = 0

    print(f"Starting training with threshold {threshold:.2f}")
    p = subprocess.Popen( cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,bufsize=1, creationflags=creationflags )

    # same code as before just with OS validation
    try:
        for line in p.stdout:
            print(line, end="")
            match = pattern.search(line)
            if match and float(match.group(1)) >= threshold:
                print(f"Mean Reward has reached {threshold:.2f}, stopping training.")


                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    p.terminate()

                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()

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
