import csv
import random
from sklearn import tree

IMPORTANT_COLUMNS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
EXAMPLE_COLUMN = 21

allData = []

allInput = []
allOutput = []

with open('CandidateSummary.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        allData.append(row)
allData.pop(0)

random.seed(1000)
random.shuffle(allData)

for row in allData:
    allOutput.append(row[EXAMPLE_COLUMN])
    rowData = []
    for colNum in IMPORTANT_COLUMNS:
        float(row[colNum])
        rowData.append(row[colNum])
    allInput.append(rowData)


setSizes = []
percentageCorrect = []
trainingPercentageCorrect = []

for setSize in range(5, len(allData), 5):
    splitIndex = int(round(setSize * 0.7) - 1)

    trainingInput = allInput[:splitIndex]
    trainingOutput = allOutput[:splitIndex]
    testInput = allInput[splitIndex:setSize]
    testOutput = allOutput[splitIndex:setSize]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainingInput, trainingOutput)


    totalCorrect = 0
    for index, inputRow in enumerate(testInput):
        if testOutput[index] == clf.predict([inputRow])[0]:
            totalCorrect += 1

    setSizes.append(setSize)
    percentageCorrect.append(float(totalCorrect) / len(testInput))

    totalCorrect = 0
    for index, inputRow in enumerate(trainingInput):
        if trainingOutput[index] == clf.predict([inputRow])[0]:
            totalCorrect += 1

    setSizes.append(setSize)
    trainingPercentageCorrect.append(float(totalCorrect) / len(trainingInput))


for size in setSizes:
    print(size)
print('================')
for percentage in percentageCorrect:
    print(percentage)
print('================')
for percentage in trainingPercentageCorrect:
    print(percentage)