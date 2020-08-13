import csv
import sys
import math


class parser:

    def csvParser(self, filename):
        self.S = []
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            count = 0
            for row in csvreader:
                if count == 0:
                    self.attributeNames = row[:-1]
                else:
                    self.S.append([int(i) for i in row])
                count += 1

        self.attributes = range(len(self.attributeNames))
        self.Sindex = range(len(self.S))
        return self


class Accuracy:
    def __init__(self, filename):
        csvParser = parser.csvParser(self, filename)
        self.data = csvParser.S

    def countAcc(self, root):

        if root == None or len(self.data) == 0:
            return 0
        count = 0
        for i in range(0, len(self.data)):
            if self.traverseTree(root, self.data[i]) == self.data[i][20]:
                count += 1
        self.accuracy = 1.0 * count / len(self.data)

        return self.accuracy

    def traverseTree(self, root, row):
        if root != None:
            if root.val == -1:
                return root.split
            if row[root.val] == 0:
                return self.traverseTree(root.left, row)
            else:
                return self.traverseTree(root.right, row)



class Node:
    def __init__(self, val):
        self.val = val
        self.split = -1
        self.left = None
        self.right = None


class DTree:

    def __init__(self, filename, heuristic):

        csvParser = parser.csvParser(self, filename)
        self.heuristic = heuristic
        self.attributeNames = csvParser.attributeNames
        self.S = csvParser.S
        self.attributes = csvParser.attributes
        self.trainingValues0 = csvParser.Sindex
        self.Sindex = list(self.trainingValues0)

        self.root = self.ID3(self.Sindex, self.attributes)

    def ID3(self, Sindex, X):

        if len(Sindex) == 0:
            return None
        root = Node(-1)
        Entropy = self.getEntropy(Sindex)
        generalvi = self.generalvi(Sindex)
        root.split = self.getlarge(Sindex)  # classification

        if Entropy == 0 or len(X) == 0:
            return root
        else:
            if self.heuristic == "1":
                bestx = self.entropy(Sindex, X, Entropy)
            else:
                bestx = self.impurity(Sindex, X, generalvi)


            if bestx == -1:
                return root
            root.val = bestx
            newAttributes = []

            for x in X:
                if x != bestx:
                    newAttributes.append(x) #all attributes but not the prev. best

            X = newAttributes
            subdata = self.split(Sindex, bestx)
            root.left = self.ID3(subdata[0][0], X)
            root.right = self.ID3(subdata[1][0], X)

            return root
    def entropy(self, Sindex, X, Entropy):

        maxInfoGain = -1
        bestx = -1

        for x in X:
            infoGain = self.infoGain(Sindex, Entropy, x)

            if infoGain > maxInfoGain:
                maxInfoGain = infoGain
                bestx = x
        return bestx
    def impurity(self, Sindex, X, vi):

        maxvigain = -1
        bextX = -1

        for x in X:
            vigain = self.VI(Sindex, vi, x)

            if vigain > maxvigain:
                maxvigain = vigain
                bextX = x

        return bextX
    def getlarge(self, S):
        size = len(S)
        if size == 1:
            return self.S[S[0]][20]
        count = 0
        for i in range(size):
            if self.S[S[i]][20] == 1:
                count += 1

        if count >= size / 2:
            return 1
        else:
            return 0
    def split(self, S, X):

        data = []
        data1 = []
        dataclass = []
        dataclass1 = []

        for i in range(len(S)):
            if self.S[S[i]][X] == 0:
                data.append(S[i])  ##inddexing
                dataclass.append(self.S[S[i]][20])
            else:
                data1.append(S[i])  ##inddexing
                dataclass1.append(self.S[S[i]][20])

        return [(data, dataclass), (data1, dataclass1)]
    def getEntropy(self, S):
        rows = len(S)

        pcount = 0
        ncount = 0
        for i in range(len(S)):
            if self.S[S[i]][20] == 1:
                pcount = pcount + 1
            else:
                ncount = pcount + 1
        if pcount == 0 or ncount == 0 or rows == 0:
            return 0

        pos = 1.0 * pcount / rows
        neg = 1 - pos

        return - (pos * math.log(pos, 2) + neg * math.log(neg, 2))
    def infoGain(self, trainingValues, Entropy, attribute):

        rows = len(trainingValues)

        subData = self.split(trainingValues, attribute)
        # print subTree
        EntropyVal0 = self.getEntropy(subData[0][0])
        EntropyVal1 = self.getEntropy(subData[1][0])

        probVal0 = 1.0 * len(subData[0][0]) / rows
        probVal1 = 1 - probVal0

        infoGain = Entropy - probVal0 * EntropyVal0 - probVal1 * EntropyVal1
        return infoGain
    def generalvi(self, S):

        rows = len(S)
        pcount = 0
        ncount = 0

        for i in range(len(S)):
            if self.S[S[i]][20] == 1:
                pcount += 1
            elif self.S[S[i]][20] == 0:
                ncount += 1

        if pcount == 0 or ncount == 0 or rows==0:
            return 0
        pos = 1.0 * pcount / rows
        neg = 1.0 * ncount / rows

        return pos * neg
    def VI(self, S, vi, attribute):

        rows = len(S)

        subTree = self.split(S, attribute)
        vI0 = self.generalvi(subTree[0][0])
        vI1 = self.generalvi(subTree[1][0])

        probVal0 = 1.0 * len(subTree[0][0]) / rows
        probVal1 = 1 - probVal0

        VI = vi - probVal0 * vI0 - probVal1 * vI1

        return VI
    def displayTree(self, root, depth, attributeNames):

        Res = ''
        if root == None:
            return ''
        if root.left == None and root.right == None:
            Res =Res+ str(root.split) + '\n'
            return Res

        currentNode = attributeNames[root.val]

        level = ''
        for i in range(0, depth):
            level += '| '
        Res =Res+ level

        if root.left != None:
            if root.left.left == None and root.left.right == None:
                Res =Res+ currentNode + "= 0 :"
            else:
                Res =Res+ currentNode + "= 0 :\n"
        Res =Res+ self.displayTree(root.left, depth + 1, attributeNames)

        Res =Res+ level
        if root.right != None:
            if root.right.left == None and root.right.right == None:
                Res =Res+ currentNode + "= 1 :"
            else:
                Res =Res+ currentNode + "= 1 :\n"
        Res =Res+ self.displayTree(root.right, depth + 1, attributeNames)

        return Res


def main():
    directory = "dataset 1/"
    #directory="dataset 2/"

    train_file = directory + str(sys.argv[1])
    validation_file = directory + sys.argv[2]
    test_file = directory + str(sys.argv[3])
    toPrint = sys.argv[4]  # yes/no
    heuristic = sys.argv[5]
    #python3 main.py training_set.csv validation_set.csv test_set.csv no 1
    decisionTree = DTree(train_file, heuristic)
    if toPrint == "yes":
        print(decisionTree.displayTree(decisionTree.root, 0, decisionTree.attributeNames))

    accuracy = Accuracy(train_file)
    accuracy.countAcc(decisionTree.root)
    print("Accuracy on ",end="")
    print(train_file)

    print(str((accuracy.accuracy) * 100) + "%")

    accuracy = Accuracy(validation_file)
    accuracy.countAcc(decisionTree.root)
    print("Accuracy on ",end="")
    print(validation_file)
    print(str((accuracy.accuracy) * 100) + "%")


    accuracy = Accuracy(test_file)
    accuracy.countAcc(decisionTree.root)
    print("Accuracy on ",end="")
    print(test_file)
    print(str((accuracy.accuracy) * 100) + "%")




if __name__ == '__main__':
    main()
