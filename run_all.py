import csv
import random
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor


def getOutputAndInputByColumnList(IMPORTANT_COLUMNS, EXAMPLE_COLUMN, fileLocation):
  allData = []
  allInput = []
  allOutput = []

  with open(fileLocation, 'rb') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      for row in reader:
          allData.append(row)
  allData.pop(0)

  random.seed(1000)
  random.shuffle(allData)

  for row in allData:
      allOutput.append(float(row[EXAMPLE_COLUMN]))
      rowData = []
      for colNum in IMPORTANT_COLUMNS:
          float(row[colNum])
          rowData.append(float(row[colNum]))
      allInput.append(rowData)
  return (allInput, allOutput)

def getOutputAndInputByColumnRange(IMPORTANT_COLUMNS_RANGE, EXAMPLE_COLUMN, fileLocation):
  allData = []
  allInput = []
  allOutput = []

  with open('SpeedDating.csv', 'rb') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      for row in reader:
          allData.append(row)
  allData.pop(0)

  random.seed(1000)
  random.shuffle(allData)

  for rowNumber, row in enumerate(allData):
      allOutput.append(float(row[EXAMPLE_COLUMN]))
      rowData = []
      for colNum in IMPORTANT_COLUMNS_RANGE:
          try:
              float(row[colNum])
              rowData.append(float(row[colNum]))
          except:
              pass
      allInput.append(rowData)
  return (allInput, allOutput)

# ==================================================================
# Modify dynamic data

datingData = getOutputAndInputByColumnRange(range(12, 82), 11, 'SpeedDating.csv')
candidateData = getOutputAndInputByColumnList([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 21, 'CandidateSummary.csv')

decisionTreeClassifier = tree.DecisionTreeClassifier()
neuralNetClassifier1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 10), random_state=1)
neuralNetClassifier2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
adaBoostClf = AdaBoostClassifier(n_estimators=10)
svmClassifier = svm.SVC()
svmClassifier2 = svm.SVC(kernel='linear')
kNeighbors2 = KNeighborsRegressor(n_neighbors=2)
kNeighbors4 = KNeighborsRegressor(n_neighbors=4)
kNeighbors8 = KNeighborsRegressor(n_neighbors=8)
kNeighbors16 = KNeighborsRegressor(n_neighbors=16)


dataSets = [candidateData, datingData]
classifiers = [svmClassifier2]

# ==================================================================

for dataIndex, dataSet in enumerate(dataSets):
  for classifierIndex, clf in enumerate(classifiers):
    startTime= datetime.now()

    setSizes = []
    percentageCorrect = []
    trainingPercentageCorrect = []

    for setSize in range(5, len(dataSet[0]), 5):
        splitIndex = int(round(setSize * 0.7) - 1)

        trainingInput = dataSet[0][:splitIndex]
        trainingOutput = dataSet[1][:splitIndex]
        testInput = dataSet[0][splitIndex:setSize]
        testOutput = dataSet[1][splitIndex:setSize]

        try:
          clf = clf.fit(trainingInput, trainingOutput)


          totalCorrect = 0
          for index, inputRow in enumerate(testInput):
              if testOutput[index] == clf.predict([inputRow])[0]:
                  totalCorrect += 1

          percentageCorrect.append(float(totalCorrect) / len(testInput))

          totalCorrect = 0
          for index, inputRow in enumerate(trainingInput):
              if trainingOutput[index] == clf.predict([inputRow])[0]:
                  totalCorrect += 1
          trainingPercentageCorrect.append(float(totalCorrect) / len(trainingInput))
        except:
          percentageCorrect.append('-')
          trainingPercentageCorrect.append('-')

        setSizes.append(setSize)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('DataIndex: {} - ClassifierIndex: {}'.format(dataIndex, classifierIndex))
    for size in setSizes:
        print(size)
    print('================')
    for percentage in percentageCorrect:
        print(percentage)
    print('================')
    for percentage in trainingPercentageCorrect:
        print(percentage)
    timeElapsed=datetime.now()-startTime
    print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')