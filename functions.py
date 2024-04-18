import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


def calcProb(currentTime, listOfNumAndTime, averageNum):
    """
    Calculates current prob of item being in store given relavent parameters.
    Takes in currentTime: current time when called,
             listOfNumAndTime: proccessed list of number and times of product sightings,
             averageNum: average number of items in stock across stores.
    Returns probability of product being in store.
    """
    daysPassed = round((currentTime - listOfNumAndTime[1])/(24 * 60 * 60))

    if listOfNumAndTime[0] == 0:
        return min(daysPassed/14, .5)

    if daysPassed == 0:
        return 1

    return min(1, max(0,(1 - daysPassed/listOfNumAndTime[0]))  +  min(daysPassed/14, .5) )

def individualFilter(itemType, currentTime, oneLocDict, averageNum):
    """
    Changes one location entry with all the usual data to a location and prob entry.
    Takes in itemType: type of item in entry,
             currentTime: current time when called,
             oneLocDict: single entry from initial data dictionary in original format,
             averageNum: average number of items in stock across stores.
    Returns new entry, as a dictionary with latlong and prob keys.
    """

    filteredInnerDict = {'latlong': oneLocDict['latlong'],
                  'prob': calcProb(currentTime, oneLocDict['stock'][itemType], averageNum)
                 }
    return filteredInnerDict

def preProccessFilterAndComputeProbs(itemType, currentTime, locationsDict):
    """
    Takes in locationsDict (only considering one entry) and proccesses each
    entry to be a location and prob entry.
    Takes in itemType: type of item to consider in entries,
             currentTime: current time when called,
             locationsDict: initial dictionary with sightings.
    Returns new dictionary, with updated entries.
    """

    # Calculate average across stores
    averageNum = 0
    for locid in locationsDict:
        averageNum += locationsDict[locid]['stock'][itemType][0]
    averageNum = averageNum/len(locationsDict.keys())

    # create new dictionary, using individualFilter fuction in each entry
    newFilteredDict = {}
    for locid in locationsDict:
        newFilteredDict[locid] = individualFilter(itemType, currentTime, locationsDict[locid], averageNum)
    return newFilteredDict

def preProccessFilterAndComputeProbsMultipleItems(itemTypes, currentTime, locationsDict):
    """
    Takes in locationsDict (considering all wanted entries) and proccesses each
    entry to be a location and prob entry, again considering all wanted entries.
    Takes in itemTypes: list with types of item to consider in entries,
             currentTime: current time when called,
             locationsDict: initial dictionary with sightings.
    Returns new dictionary, with updated entries.
    """

    # Initialize base dictionary
    newCombined = {}
    for locid in locationsDict:
        newCombined[locid] = {'latlong': locationsDict[locid]['latlong'],
                              'prob': 1
                             }

    # filter on each relevent item type
    filtered = [(preProccessFilterAndComputeProbs(itemType, currentTime, locationsDict))  for itemType in itemTypes]

    # combine
    for individualFiltered in filtered:
        for locid in individualFiltered:
            newCombined[locid]['prob'] *= individualFiltered[locid]['prob']

    return newCombined

def swapPositions(l, pos1, pos2):
    """
    Helper funtion to swap two positions in a list.
    Takes in a list and two indices in the list.
    """
    l2 = l.copy()
    l2[pos1], l2[pos2] = l2[pos2], l2[pos1]
    return l2

def calcPathTimeInPath(path, preProccessed, currentNodeId):
    """
    Given a path, our proccessed dictionary and current spot in path, computes
    expected time till end of search.
    Takes in path: our current candidate path (a list)
             preProccessed: dictionary proccessed by preProccessFilterAndComputeProbsMultipleItems
             currentNodeId: spot in path
    Returns time left in path.
    """

    probLeave = (1 - preProccessed[currentNodeId]['prob']) # prob of finding product
    #distance to next node
    distToNext = np.linalg.norm( np.array(preProccessed[path[0]]['latlong']) - np.array(preProccessed[currentNodeId]['latlong'])  )

    if len(path) == 1:
        return probLeave * distToNext

    # new distance if you don't find product in current store
    newDist = distToNext + calcPathTimeInPath(path[1:], preProccessed, path[0])
    # return expected time
    return probLeave * newDist

def calcPathTime(path, preProccessed, startingLocation):
    """
    Given a path, our proccessed dictionary and starting location, computes
    expected time till end of search.
    Takes in path: our current candidate path (a list)
             preProccessed: dictionary proccessed by preProccessFilterAndComputeProbsMultipleItems
             startingLocation: startingLocation in path
    Returns expected time to complete path.
    """
    distToFirstStore = np.linalg.norm(np.array(startingLocation) - np.array(preProccessed[path[0]]['latlong']))
    return distToFirstStore + calcPathTimeInPath(path[1:], preProccessed, path[0])

def minDistGradDescent(itemTypes, currentTime, locationsDict, startingLocation):
    """
    Performs gradient descent on paths, giving us our best path.
    Takes in itemTypes: list with types of items to consider in entries,
             currentTime: current time when called,
             locationsDict: initial dictionary with sightings,
             startingLocation: where search starts from.
    Returns best found path, expected distance for this path, and a list of
    tried candidate paths in order.
    """
    preProccessed = preProccessFilterAndComputeProbsMultipleItems(itemTypes, currentTime, locationsDict)

    starting = list(preProccessed.keys())

    # For animation later
    toAnimation = [starting]

    diff = 1
    i = 0
    # Perform gradient descent
    while i<50 and diff != 0:
        i += 1
        bestFound, expDistToGo, diff = minPathOneStep(starting, preProccessed, startingLocation)
        starting = bestFound
        toAnimation.append(starting)

    return  bestFound, expDistToGo, toAnimation

def minPathOneStep(starting, preProccessed, startingLocation):
    """
    Perform one step of gradient descent on paths, using all possible swaps.
    Takes in starting: current candidate path,
             preProccessed: dictionary proccessed by preProccessFilterAndComputeProbsMultipleItems,
             startingLocation: where search starts from.
    Returns new best found path, current min expected distance, and difference
    in expected time from starting to new best found path.
    """
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
    """
    Help function: Draw scatter plot with locationsDict and annotate it (for itemType).
    """
    for locid in locationsDict:
        plt.scatter(locationsDict[locid]['latlong'][0], locationsDict[locid]['latlong'][1])
        plt.annotate(locationsDict[locid]['stock'][itemType][0], (locationsDict[locid]['latlong'][0], locationsDict[locid]['latlong'][1]),
                     xytext=(5, 5),
                     textcoords='offset points')

    plt.show()
    return None

def drawProbs(proccessedDict):
    """
    Help function: Draw scatter plot with proccessedDict and annotate it with entry probabilities.
    """
    for locid in proccessedDict:
        plt.scatter(proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1])
        plt.annotate(round(proccessedDict[locid]['prob'],3), (proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1]),
                     xytext=(5, 5),
                     textcoords='offset points'
                    )
    plt.show()
    return None

def drawPath(paths, startingPoint, proccessedDict):
    """
    This creates the animation depicting our algorithm choosing its path.
    Takes in paths: paths we searched through in minDistGradDescent as list of paths.
             startingPoint: starting point to begin search.
             proccessedDict: dictionary proccessed by preProccessFilterAndComputeProbsMultipleItems.
    Returns None, but saves animation to optimalPath.gif
    """
    fig, ax = plt.subplots()

    # annotate
    for locid in proccessedDict:
        ax.scatter(proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1])
        ax.annotate(round(proccessedDict[locid]['prob'],3), (proccessedDict[locid]['latlong'][0], proccessedDict[locid]['latlong'][1]),
                     xytext=(5, 5),
                     textcoords='offset points'
                    )

    # Update each step in animation
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

    # Create animation
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(paths), interval=len(paths)*3, repeat=False)
    writergif = animation.PillowWriter(fps=4)
    ani.save('optimalPath.gif', writer=writergif)
    plt.show()
    return None
