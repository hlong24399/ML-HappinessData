import pandas as pd
# from matplotlib import pyplot as plt
from scipy.stats import pearsonr as pc
from sklearn.model_selection import train_test_split
import math as m


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


df = pd.read_csv("HappinessData.csv")

# Fill missing data
fill_all_missing(df)

# Get label column
label_column_title = 'Unhappy/Happy'
label_column = df.pop(label_column_title)

# Added the label column to the end of the dataset
df.insert(len(df.columns), label_column_title, label_column)


# Pearsons Correlation
cors = []
for (columnName, columnData) in df.iteritems():
  cors.append(pc(columnData, label_column)[0])
cors.pop()


# split dataset - THIS SHUFFLE AGAIN ONCE COMPILE
train, test = train_test_split(df, test_size=0.2)

# print(train.loc[0].index[4])
# print(train.loc[0][1])


# This is a test scenario for distances
# row0 = train.loc[0]
# distances = []
# for i in range(0, len(train)):
#   distances.append(euclidean_distance(row0, train.iloc[i]))

# calculate the Euclidean distance
def euclidean_distance(row1 : pd.Series, row2 : pd.Series):
  distance = 0.0
  # print(row1)
  for a in range(0,len(row1)-1):
    distance+=((row1[a] - row2[a])**2)
  # print(distance)
  # print(m.sqrt(distance))
  return m.sqrt(distance)  

def get_neighbors(train : pd.DataFrame, test_row, k):
  distances = []
  for i in range(0, len(train)):
    dist = euclidean_distance(train.iloc[i], test_row)
    distances.append((dist, train.  iloc[i][6]))
  distances.sort()
  # print(distances[0:3][0], "and", distances[0:3][1])
  return distances[0:k]

# Get the desired number of neighbors and predict
def predict(train_set, new_rec, k=5):
  neighbors = get_neighbors(train_set, new_rec, k)
  categories = [0,0]
  for neighbor in neighbors:
    # print(neighbor[1])
    categories[int(neighbor[1])]+=1
  # print(categories.index(max(categories)))
  return categories.index(max(categories))

# my_neighbor = get_neighbors(train, test.iloc[3], 4)

prediction = predict(train, test.iloc[3])




# # Scikit-learn version
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(label_column, train)
