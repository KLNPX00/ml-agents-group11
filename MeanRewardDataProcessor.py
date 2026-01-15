import pandas as pd

STEP_BIN = 60_000
df = pd.read_excel("Total_data_question_2.xlsx")

df = df.rename(columns={
    df.columns[0]: "nrSteps",
    df.columns[1]: "meanReward"      
})

#ensure numeric
df["nrSteps"] = pd.to_numeric(df["nrSteps"], errors="coerce")
df["meanReward"] = pd.to_numeric(df["meanReward"], errors="coerce")

#here when we notice a new 60K we make a new runNr
df["runNr"] = (df["nrSteps"].diff() < 0).cumsum() + 1

#to add time units we divide the current nr of steps by 60k
df["timeUnit"] = (df["nrSteps"] // STEP_BIN).astype(int)
#order the columns
df = df[["runNr", "nrSteps", "timeUnit", "meanReward"]]

#sort
df = df.sort_values(["runNr", "nrSteps"]).reset_index(drop=True)

#here we create a new file we can put either excel(.xsls) or .csv
df.to_csv("MeanReward_training_data.xlsx", index=False)

