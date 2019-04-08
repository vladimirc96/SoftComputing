# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import math
import cv2
from helpers import *


def calculate_distance(x1, y1, x2, y2, x0, y0):
    denominator = (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1

    if denominator > 0:
        denominator = denominator * (-1)
    else:
        denominator = denominator * (-1)

    numerator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return denominator / numerator


class CentroidTracker():
    def __init__(self, maxDisappeared=2):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.added = OrderedDict()
        self.type = ""
        self.total_sum = 0

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
    def getAdded(self):
        return self.added

    def setType(self,trackerType):
        self.type = trackerType
    def getSum(self):
        return self.total_sum
    # u tracker se objekat registruje tek kada prodje ispod linije
    # cim se registruje vrsi se dodavanje/oduzimanje u zavisnosti od vrste trackere i vrsi se dalje pracenje objekta
    def register(self, centroid, boxes, frame, loaded_model):
        regions = []
        alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        img = image_bin(image_gray(blur))
        region = img[boxes[1] - 4:boxes[1] - 4 + boxes[3]-boxes[1] + 8, boxes[0] - 8:boxes[0] - 8 + boxes[2]-boxes[0] + 16]
        resized = resize_region(region)
        scaled = scale_to_range(resized)
        vector = matrix_to_vector(scaled)
        regions.append(vector)
        result_ann = loaded_model.predict(np.array(regions, np.float32))
        if(self.type == "BLUE"):
            temp = display_result(result_ann, alphabet)
            self.total_sum += temp[0]
        else:
            temp = display_result(result_ann, alphabet)
            self.total_sum += temp[0]
        # when registering an object we use the next available object
        # ID to store the centroid
        # ako je registrovan to znaci da je prosao ipsod linije - dodat je
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def getAdded(self):
        return self.added

    def update(self, rects,x1,y1,x2,y2, frame, loaded_model):
        # proveri da li su se javile konture
        if len(rects) == 0:
            remove = []
            # proveri sve koje vec pratis i vidi da li su ispali iz opsega
            # ili su prekoracili broj frejmova za koje su "nestali"
            for objectID in self.objects.keys():
                centroid = self.objects[objectID]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    remove.append(objectID)

                if centroid[1] > y1 + 5 and centroid[0] < x1 - 5 and centroid[0] > x2 + 5:
                    if objectID in remove:
                        continue
                    else:
                        remove.append(objectID)
            for r in remove:
                self.deregister(r)
            # izadji odma posto nema novih kontura
            return self.objects

        # inicijalizuj input centroide za trenutni frejm
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        boxes = []
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            boxes.append(rects[i])

        # ako ne pratimo objekte uzmi input centroida i registruj ih
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], boxes[i], frame, loaded_model)

        # u suprotnom, pratimo vec objekte i treba da ih povezemo sa
        # odgovarajucim centroidima iz inputa
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # izracunaj daljinu izmedju objekata koje pratis i novih pozicija pojedinacno - input centroida
            # koji predstavljaju pomeraje u odnosu na proslu poziciju
            # D je niz nizova, svaki niz sadrzi daljine izmedju objekta koji se prati i novih pozicija
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # D.min(axis=1) vraca minimalno rastojanje izmedju objekta koji se prati i novih pozicija
            # sortiraj kako bi dobio id-eve
            rows = D.min(axis=1).argsort()

            # cols sadrzi informaciju o tome na kom indeksu se nalazi nova pozicija
            # za svaki objekat iz input centroida
            # uradi se argmin() za odgovarajuce id-eve (indexe) iz rows
            cols = D.argmin(axis=1)[rows]

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
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # ako je broj objects veci od broja inputa mora se proveriti da li je
            # object nestao - moze se desiti ako se preklope brojevi ili ako prelazi preko linije
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                remove = []
                for row in unusedRows:
                    objectID = objectIDs[row]
                    centroid = self.objects[objectID]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

                    if centroid[1] > y1 + 5 and centroid[0] < x1 - 5 and centroid[0] > x2 + 5:
                        if objectID in remove:
                            continue
                        else:
                            remove.append(objectID)
                    for r in remove:
                        self.deregister(r)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], boxes[col], frame, loaded_model)

        # return the set of trackable objects
        return self.objects


