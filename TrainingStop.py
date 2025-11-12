import subprocess, signal, re, time # standard library imports

THRESHOLD = 4.632   #treshold we want to test. Right now it is only mean reward. We need to decide on how ot measure treshold
# if you change the treshold, change the run id name in CDM too
CMD=["mlagents-learn","config/ppo/PushBlock.yaml","--run-id","treshold4.632Run"] # commands needed to run the training

# we use .re to look for the pattern described under so that we can track the Mean reward
pattern=re.compile(r"Mean Reward: (\d+\.\d+)") #  reads format "Mean reward: 4.521" for example

p=subprocess.Popen(CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True, bufsize=1,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) #line that runs the training using the CMD commands 

try:
    for line in p.stdout:
        print(line, end="") # makes sure there are no double spaces
        match = pattern.search(line)
        if match and float(match.group(1)) >= THRESHOLD: # checks if treshold ( in this case mean reward) is smaller than current mean reward
            print(f"Mean Reward has reached the {THRESHOLD}, stopping training")
            p.send_signal(signal.CTRL_BREAK_EVENT) #signal that works as CTRL + C
            try:
                p.wait(timeout=30) # wait to make sure it has time to save data
            except subprocess.TimeoutExpired:
                p.terminate() # terminate it if it takes too long
            break
 # safety net in case you want to terminate the program with CTRL + C, it will send the termination to the subprogram(child)
except KeyboardInterrupt:
    p.send_signal(signal.CTRL_BREAK_EVENT) 