# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import math

def calculate_distance(x1, y1, x2, y2, x0, y0):
    denominator = (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1

    if denominator > 0:
        denominator = denominator * (-1)
    else:
        denominator = denominator * (-1)

    numerator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return denominator / numerator


class CentroidTracker():
    def __init__(self, maxDisappeared=3):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.added = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
    def getAdded(self):
        return self.added

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.added[self.nextObjectID] = False
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        #del self.disappeared[objectID]

    def getAdded(self):
        return self.added

    def update(self, rects,x1,y1,x2,y2):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            remove = []
            # proveri sve koje vec pratis i vidi da li su ispali iz opsega
            for objectID in self.objects.keys():
                centroid = self.objects[objectID]
                #self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    remove.append(objectID)

                if centroid[1] > y1 and centroid[0] < x1 and centroid[0] > x2:
                    if objectID in remove:
                        continue
                    else:
                        remove.append(objectID)
                #     if centroid[1] > y1 + 40 or centroid[0] > x2 + 30:
                #         remove.append(objectID)
                for r in remove:
                    self.deregister(r)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)


        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            #grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                # row je ID a col je indeks pozizicije nove vrednosti iz inputa
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            # ako je u un
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # ako je broj objects veci od broja inputa mora se proveriti da li je
            # object nestao - moze se desiti ako se preklope brojevi ili ako prelazi preko linije
            # postaviti uslov ako prelazi liniju da ima fore 3 frejma ?
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                remove = []
                for row in unusedRows:
                    objectID = objectIDs[row]
                    centroid = self.objects[objectID]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] >= self.maxDisappeared:
                        remove.append(objectID)

                    if centroid[1] > y1 + 5 and centroid[0] < x1 and centroid[0] > x2 + 5:
                        if objectID in remove:
                            continue
                        else:
                            remove.append(objectID)

                    for r in remove:
                        if r in self.objects.keys():
                            self.deregister(r)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects


