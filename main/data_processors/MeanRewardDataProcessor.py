import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))#get the path of the current folder
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))#go back one step to reach main and go to the data folder
INPUT= os.path.join(DATA_DIR, "Total_data_question_2.xlsx")#file we get the data from(change if needed)
OUTPUT= os.path.join(DATA_DIR, "MeanReward_training_data.csv")#output file(change if needed)

STEP_BIN = 60_000
df = pd.read_excel(INPUT)

df = df.rename(columns={
    df.columns[0]: "nrSteps",
    df.columns[1]: "meanReward"      
})

#ensure numeric
df["nrSteps"] = pd.to_numeric(df["nrSteps"], errors="coerce")
df["meanReward"] = pd.to_numeric(df["meanReward"], errors="coerce")

#here when we notice a new 60K we make a new runNr
df["runNr"]= (df["nrSteps"].diff() <0).cumsum() + 1

#to add time units we divide the current nr of steps by 60k
df["timeUnit"] = (df["nrSteps"] // STEP_BIN).astype(int)
#order the columns
df = df[["runNr","nrSteps", "timeUnit", "meanReward"]]

#sort
df = df.sort_values(["runNr", "nrSteps"]).reset_index(drop=True)

#here we create a new file we can put either excel(.xsls) or .csv
df.to_csv(OUTPUT, index=False)

