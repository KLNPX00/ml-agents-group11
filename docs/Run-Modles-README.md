# Run Models

Prerequisites: pip install pandas

pip install matplotlib

pip install scikit-learn

### For the following models you can run them in the same manner.

-   GradientBoostingDecisionTrees
-   RandomForestRegressor
-   KNN

###### To run the models:

-   python "filename".py
-   Example: python GradientBoostingDecisionTrees.py
-   It may require the full path of the file instead of just the filename.

This will run the models with a set parameter for a mean reward of 4 and the output will be displayed in the terminal.

### For the following models you will need to change the parameters in the command line.

-   CatBoost
-   LightGBM

###### To run the models:

* CatBoost: python CatBoost.py --metric ram_peak_python --dataset dataset-1000.csv
* LightGBM: python LightGBM.py --metric ram_peak_python --dataset dataset-1000.csv

"ram_peak_python" can be replaced with these options:
- cpu_avg_unity
- cpu_peak_unity
- ram_avg_unity
- ram_peak_unity
- cpu_avg_python
- cpu_peak_python
- ram_avg_python

dataset-1000.csv is just a name of the file with data; any such file under data directory would work too.
