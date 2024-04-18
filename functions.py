import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


def calcProb(currentTime, listOfNumAndTime, averageNum):
    """
    Calculates current prob of item in pharmacy given relavent parameters.
    """
    daysPassed = round((currentTime - listOfNumAndTime[1])/(24 * 60 * 60))

    if listOfNumAndTime[0] == 0:
        return min(daysPassed/14, .5)

    if daysPassed == 0:
        return 1

    return min(1, max(0,(1 - daysPassed/listOfNumAndTime[0]))  +  min(daysPassed/14, .5) )

def individualFilter(itemType, currentTime, oneLocDict, averageNum):
    filteredInnerDict = {'latlong': oneLocDict['latlong'],
                  'prob': calcProb(currentTime, oneLocDict['stock'][itemType], averageNum)
                 }
    return filteredInnerDict

def preProccessFilterAndComputeProbs(itemType, currentTime, locationsDict):
    averageNum = 0
    for locid in locationsDict:
        averageNum += locationsDict[locid]['stock'][itemType][0]
    averageNum = averageNum/len(locationsDict.keys())

    newFilteredDict = {}
    for locid in locationsDict:
        newFilteredDict[locid] = individualFilter(itemType, currentTime, locationsDict[locid], averageNum)
    return newFilteredDict

def preProccessFilterAndComputeProbsMultipleItems(itemTypes, currentTime, locationsDict):
    newCombined = {}
    for locid in locationsDict:
        newCombined[locid] = {'latlong': locationsDict[locid]['latlong'],
                              'prob': 1
                             }

    filtered = [(preProccessFilterAndComputeProbs(itemType, currentTime, locationsDict))  for itemType in itemTypes]

    for individualFiltered in filtered:
        for locid in individualFiltered:
            newCombined[locid]['prob'] *= individualFiltered[locid]['prob']

    return newCombined

def swapPositions(l, pos1, pos2):
    l2 = l.copy()
    l2[pos1], l2[pos2] = l2[pos2], l2[pos1]
    return l2

def calcPathTimeInPath(path, preProccessed, currentNodeId):
    probLeave = (1 - preProccessed[currentNodeId]['prob'])
    distToNext = np.linalg.norm( np.array(preProccessed[path[0]]['latlong']) - np.array(preProccessed[currentNodeId]['latlong'])  )

    if len(path) == 1:
        return probLeave * distToNext

    newDist = distToNext + calcPathTimeInPath(path[1:], preProccessed, path[0])
    return probLeave * newDist

def calcPathTime(path, preProccessed, startingLocation):
    distToFirstStore = np.linalg.norm(np.array(startingLocation) - np.array(preProccessed[path[0]]['latlong']))
    return distToFirstStore + calcPathTimeInPath(path[1:], preProccessed, path[0])

def minDistGradDescent(itemTypes, currentTime, locationsDict, startingLocation):
    preProccessed = preProccessFilterAndComputeProbsMultipleItems(itemTypes, currentTime, locationsDict)

    starting = list(preProccessed.keys())
    toAnimation = [starting]

    diff = 1
    i = 0
    while i<50 and diff != 0:
        i += 1
        bestFound, expDistToGo, diff = minPathOneStep(starting, preProccessed, startingLocation)
        starting = bestFound
        toAnimation.append(starting)

    return  bestFound, expDistToGo, toAnimation

def minPathOneStep(starting, preProccessed, startingLocation):
    currentMinExpTime = calcPathTime(starting, preProccessed, startingLocation)
    firstTime = currentMinExpTime
    currentMinPath = starting

    for firstInx in range(len(starting)):
        for endingInx in range(len(starting))[firstInx+1:]:
            candList = swapPositions(starting, firstInx, endingInx)
            candExpTime = calcPathTime(candList, preProccessed, startingLocation)

            if candExpTime < currentMinExpTime:
                currentMinPath = candList
                currentMinExpTime = candExpTime

    return currentMinPath, currentMinExpTime, firstTime - currentMinExpTime

def drawState(locationsDict, itemType):
    for locid in locationsDict:
        plt.scatter(locationsDict[locid]['latlong'][0], locationsDict[locid]['latlong'][1])
        plt.annotate(locationsDict[locid]['stock'][itemType][0], (locationsDict[locid]['latlong'][0], locationsDict[locid]['latlong'][1]),
                     xytext=(5, 5),
                     textcoords='offset points')

    plt.show()
    return None

def drawProbs(proccessedDict):
    for locid in proccessedDict:
        plt.scatter(proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1])
        plt.annotate(round(proccessedDict[locid]['prob'],3), (proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1]),
                     xytext=(5, 5),
                     textcoords='offset points'
                    )
    plt.show()
    return None

def drawPath(paths, startingPoint, proccessedDict):
    fig, ax = plt.subplots()

    # annotate
    for locid in proccessedDict:
        ax.scatter(proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1])
        ax.annotate(round(proccessedDict[locid]['prob'],3), (proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1]),
                     xytext=(5, 5),
                     textcoords='offset points'
                    )

    def update(i):
        locs = [startingPoint]
        for node in paths[i]:
            locs.append(proccessedDict[node]['latlong'])

        ax.clear()
        for locid in proccessedDict:
            ax.scatter(proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1])
            ax.annotate(round(proccessedDict[locid]['prob'],3), (proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1]),
                         xytext=(5, 5),
                         textcoords='offset points'
                        )
        for locIndex in range(len(locs) - 1):
            data = [(locs[locIndex][0], locs[locIndex+1][0]),
                    (locs[locIndex][1], locs[locIndex+1][1]),
                    'b'
                   ]
            ax.plot(*data, alpha=0.75/(np.sqrt(locIndex + 1)+0))
        return fig, ax

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(paths), interval=len(paths)*3, repeat=False)
    writergif = animation.PillowWriter(fps=4)
    ani.save('optimalPath.gif', writer=writergif)
    plt.show()
    return None
