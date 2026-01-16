import subprocess
import signal
import re
import os
import time, math
import sys


#target threshold
#once it is reached the program will end and start with threshold +0.05
THRESHOLD = 4.00
TIMESTAMP = math.floor(time.time()) #timestamp used to make training files unique and avoid conflicts
STARTPORT=6006 #starting port
pattern = re.compile(r"Mean Reward: (\d+\.\d+)")#pattern used to check if we reached the threshold
PYTHON = sys.executable # ensures we are using the correct python(the one that has mlagents)
DIR = os.path.dirname(os.path.abspath(__file__))
EXPORTER =os.path.abspath(os.path.join(DIR, "..", "data_processors","export_tb_to_excel.py"))#file location for the export to excel file
CONFIG_FILE = os.path.abspath(os.path.join(DIR, "..", "..","config","ppo","PushBlock.yaml"))#location of the config file
ENV_PATH = os.path.abspath(os.path.join(DIR, "..", "..", "build", "UnityEnvironment.exe"))#path for the unity environment
WIN_CMD= [PYTHON,"-m","mlagents.trainers.learn",CONFIG_FILE,f"--env={ENV_PATH}","--run-id=threshold_{THRESHOLD}_t_{TIMESTAMP}","--no-graphics","--base-port={STARTPORT}"]
UNIX_CMD = [PYTHON,"-m","mlagents.trainers.learn",CONFIG_FILE,f"--env={ENV_PATH}","--run-id=threshold_{THRESHOLD}_t_{TIMESTAMP}","--no-graphics","--base-port={STARTPORT}"]

# recursive methods
def run_training(threshold,port):
    # checks OS env
    now = math.floor(time.time())#restarts the training back to the minimum threshold
    if threshold >= 4.95:
        threshold = 4.3

    if port == 6006:
        port = 6007
    else: port=6006

    if os.name == "nt":#if its nt, it is windows else it is unix based
        cmd = [arg.format(THRESHOLD=f"{threshold:.2f}",TIMESTAMP=now,STARTPORT=port) for arg in WIN_CMD]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        cmd = [arg.format(THRESHOLD=f"{threshold:.2f}",TIMESTAMP=now,STARTPORT=port) for arg in UNIX_CMD]
        preexec_fn = os.setsid

    print(f"Starting training with threshold {threshold:.2f}, timestamp: {now}")
    if os.name =="nt":
        p = subprocess.Popen( cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,bufsize=1, creationflags=creationflags )
    else :
        p = subprocess.Popen( cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,bufsize=1, preexec_fn = os.setsid )

    #here we check if we reached the threshold. If we did we send a signal to stop the training.
    #if the signal does not work we kill the processes
    try:
        for line in p.stdout:
            print(line, end="")
            match =pattern.search(line)
            if match and float(match.group(1)) >= threshold:
                print(f"Mean Reward has reached {threshold:.2f}, stopping training. timestamp: {now}.")

                if os.name =="nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)

                try:
                    p.wait(timeout=30)#wait 30 seconds to make sure the program has time to finish
                except subprocess.TimeoutExpired:
                    p.kill()

                print(f"Exporting metrics for threshold {threshold:.2f}...")
                subprocess.run(#we run the code that exports it into an excel file that will be processed further
                    [sys.executable, EXPORTER, f"{threshold:.2f}", f"{now}"],check=False)
                print(f"Excel file saved for threshold {threshold:.2f}")
                time.sleep(5)#added time between trainings if we want to stop the training manually without issues
                run_training(threshold + 0.05,port+1)

                return  # makes sure we exist the curr run to avoid crazy nesting

    #another measure in case stopping the program does not work
    except KeyboardInterrupt:
        if os.name =="nt":
            p.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            p.terminate()
        p.wait()

run_training(THRESHOLD,STARTPORT)
