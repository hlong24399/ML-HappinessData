import csv
import pandas as pd
from matplotlib import pyplot as plt

# Index(['Unhappy/Happy', 'City Services Availability', 'Housing Cost',
#        'Quality of schools', 'Community trust in local police',
#        'Community Maintenance', 'Availability of Community Room '])

# Check which columns are null
def find_missing(df):
  for (columnName, columnData) in df.iteritems():
    if (columnData.isnull().values.any() == True):
      print(columnName, " : X ")
    else:
      print(columnName, " : V ")
  print(" ---------------------------------------- ")


def fill_all_missing(df):
  for (columnName, columnData) in df.iteritems():
    if (columnData.isnull().values.any() == True):
      median = int(df[columnName].median())
      df[columnName].fillna(median, inplace=True)
  print(" ---------------------------------------- ")

# Create the edited dataset
with open('HappinessData.csv', 'r') as f:
  firstline = ''.join(str(l) for l in f.readline())
  with open('HappinessDataEdited.csv', 'w') as nf:
    # next(f)
    for line in f:
      nf.write(line)
    nf.write(firstline)

df = pd.read_csv("HappinessData.csv")
# find_missing(df)
fill_all_missing(df)
# find_missing(df)












