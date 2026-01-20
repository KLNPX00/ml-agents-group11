Prerequisites:

Download and install Unity Hub. Create a new project by selecting the folder icon. Choose the Project folder from this repository.

Download and install Anaconda from the following link:
https://www.anaconda.com/docs/getting-started/miniconda/main

## Installing the packages
After, open the Anaconda Prompt and run the following commands:

conda create -n mlagents python=3.10.12

conda activate mlagents

pip install mlagents

conda install numpy=1.23.5

pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cpu

## After installing the packages

### Instructions for headless execution:

• Open the Project in the Unity Hub Editor (put Push Block into the Scene as a normal
training would run).

• In Unity, go to File (top left), and then select Build Settings.

• In Build Settings, select the platform (it will be preselected at Windows, Mac, Linux
which is the option we need).

• In the same pop-up window, select Add Open Scenes to get the Push Block game
registered into the headless training.

• Uncheck Development Build (usually it is unchecked by default)

• Click Build

• In the window that opens, select the folder you want to place this Executable file (I put it
in a new build folder in the existing project directory).
Now, the Executable file has been created successfully. You need to follow the next steps to pass
two parameters that will enable the headless training. Please use a fresh command line for this
(not Anaconda that the training backend runs on).

• Go to the directory build (or any other name you gave to it).

• Run the command .\UnityEnvironment.exe -batchmode -nographics

• If errors are triggered, please check that the file name is correct, the directory is correct,
the extension is correct (it should be .x86_64 in Linux), and the dash parameters are
correct.
Now, go back to your Anaconda command line.

• Run the command mlagents-learn config/ppo/PushBlock.yaml –
env=./build/UnityEnvironment.exe –run-id=headless1
Now, you should be able to execute the ML-agents training process using headless execution.
Additional Commands (if you run with a pre-existing run id):

• Use --resume to start the training from the point you left. The training will continue from
the exact same point.

• Use --force to overwrite the existing data. This clears the training log file and fills it up
with fresh data.

