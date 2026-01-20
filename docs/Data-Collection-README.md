# Data Collection
To run the automated data collection you need some prerequisites.

## Prerequisites
Within the mlagents environment in the conda terminal (please refer to the installation guide) you need to install 
the following packages:

pip install pandas 

pip install tensorboard

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

chmod +x run_training.sh
chmod +x run_all_rounds.sh
./run_all_rounds.sh
//ask nikolaos for more info
