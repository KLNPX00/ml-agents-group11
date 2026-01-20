# Data Collection

## Prerequisites
Within the mlagents environment in the conda terminal (please refer to the installation guide) you need to install 
the following packages:

* pip install pandas 
* pip install tensorboard 
* pip install psutil

You will also need gitBash to run the bash script. For that download and install it from the following link:
https://git-scm.com/downloads and choose your operating system.

## Running the script
### TrainingStop.py
To run the TrainingStop.py script, please run the following command:
python main\training_files\TrainingStop.py

This will start the data collection. The data collected will be only of the mean reward and the step at which is achieved.
It will be saved in the folder "main\data\rawMeanReward_data"

### run_all_rounds.sh
You need to cd into the folder "\main\training_files\linux_setup" using the follwong command:

cd "you will need to add the previous part of the whole directory"\main\training_files\linux_setup
run the following commands:

* chmod +x run_training.sh
* chmod +x run_all_rounds.sh
* ./run_all_rounds.sh

#### For the purposes of your experimentation we have deleted some of the "done" files in round 1.If you want to see how it would look after it is done training just look at round 2
#### If you wish to run the script on your linux machine please open the run_training.sh file and comment out the indicated line in the code.
