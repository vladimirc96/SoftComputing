from scipy.spatial import distance as dist
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from keras.models import model_from_json
from centroidtracker import CentroidTracker
from helpers import *
import math

added = []
id = 0


def findLines(mask,frame):
    #er = erode(img)
    #dil = erode(dilate(er))
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(mask, 100, 300, apertureSize=5)
    plt.imshow(mask,'gray')
    plt.show()
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,maxLineGap=15,minLineLength=15)
    points = []
    for x1, y1, x2, y2 in lines[0]:
        points.append(x1)
        points.append(y1)
        points.append(x2)
        points.append(y2)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (x1,y1), 5, (200, 0, 0), 2)
        cv2.circle(frame, (x2,y2), 5, (200, 0, 0), 2)
        plt.imshow(frame)
        plt.show()

    return points



#racuna daljinu konture od linije
def calculate_distance(x1, y1, x2, y2, x0, y0):
    denominator = (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1

    if denominator > 0:
        denominator = denominator * (-1)
    else:
        denominator = denominator * (-1)

    numerator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return denominator / numerator

def process(contours, frame, numbers,result_ann, alphabet):
    i = -1
    added = ct.getAdded()
    #print(len(contours))
    #print(len(numbers))
    for contour in contours:
        i += 1
        coord = [] #lista X i Y
        x, y, w, h = cv2.boundingRect(contour)
        box = cv2.boundingRect(contour)
        box = (x, y, x+w, y+h)
        X = int((x + x + w) / 2.0)
        Y = int((y + y + h) / 2.0)
        if Y < blue_points[1] and X > blue_points[0] and X < blue_points[2]:
            rects.append(box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    distance_blue = calculate_distance(blue_points[0], blue_points[1], blue_points[2], blue_points[3], X, Y)
    # proveri da li je dovoljno udaljen, ako jeste prodji kroz sve objects i uporedi centroide sa X,Y - ako su jednaki onda za taj object stavi da je dodat
    objects = ct.update(rects,blue_points[0],blue_points[1],blue_points[2],blue_points[3])


    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        dis = calculate_distance(blue_points[0],blue_points[1],blue_points[2],blue_points[3],centroid[0],centroid[1])
        tex = "DIS {}".format(round(dis))
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)
    cv2.waitKey(50)
    # # key = cv2.waitKey(1) & 0xFF
    # objects = ct.update(rects)
    # added = ct.getAdded()
    # for (objectID, centroid) in objects.items():
    #     if centroid[1] <= blue_points[1] and centroid[1] >= blue_points[3] - 10 and centroid[0] <= blue_points[2] and centroid[0] >= blue_points[0]:
    #         distance_blue = calculate_distance(blue_points[0], blue_points[1], blue_points[2], blue_points[3], centroid[0], centroid[1])
    #         if distance_blue <= -25 and distance_blue >= -40:
    #             if added[objectID] == False:
    #                 added[objectID] = True
    #                 #cv2.circle(frame, (x, y), 5, (200, 0, 0), 2)
    #                 plt.imshow(frame)
    #                 plt.show()
    #                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #                 plt.imshow(frame)
    #                 plt.show()
    #                 blur = cv2.GaussianBlur(frame, (5, 5), 0)
    #                 img = image_bin(image_gray(blur))
    #                 region = img[y - 4:y - 4 + h + 8, x - 8:x - 8 + w + 16]
    #                 plt.imshow(resize_region(region),'gray')
    #                 plt.show()
    #                 print("*********************************SABERI************************************* ", display_result(result_ann, alphabet)[i])
    #                 print("UDALJENOST ", display_result(result_ann, alphabet)[i], "OD PLAVE: ", distance_blue)
    #                 #                 print("X:", x,"Y: ",y)
    #                 #                 print("PLAVA X1: ", blue_points[0],"PLAVA X2:", blue_points[2])
    #                 #                 print("PLAVA Y1: ", blue_points[1],"PLAVA Y2:", blue_points[3])
    #                 #                 plt.imshow(numbers[i], 'gray')
    #                 plt.show()
    #                 print("*********************************SABERI************************************* ")







#         if y <= green_points[1] and y >= green_points[3] - 10 and x <= green_points[2] and x >= green_points[0]:
#             cv2.circle(frame, (x,y), 5, (200, 0, 0), 2)
#             distance_green = calculate_distance(green_points[0],green_points[1],green_points[2],green_points[3],x,y)
#             print("UDALJENOST ", display_result(result_ann,alphabet)[i], "OD ZELENE: ",distance_green)
#             plt.imshow(numbers[i],'gray')
#             plt.show()



def tracker(contours, frame, ids, objects):
    process_contours = []
    i = 0
    for contour in contours:
        # x, y - gornji levi ugao konture
        # w, h - sirina i visina konture
        x, y, w, h = cv2.boundingRect(contour)
        if(x > blue_points[0] - 10 and x < blue_points[2] + 10 and y > blue_points[3] - 10 and y < blue_points[1] + 10):
            distance_blue = calculate_distance(blue_points[0], blue_points[1], blue_points[2], blue_points[3], x, y)
            # ako je daljina od linije negativna to znaci da je ispod linije broj prosao
            if distance_blue > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                process_contours.append(contour)
                update(process_contours)

    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        dis = calculate_distance(blue_points[0],blue_points[1],blue_points[2],blue_points[3],centroid[0],centroid[1])
        tex = "DIS {}".format(round(dis))
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def update(process_contours):

    # ako je duizina nula onda znaci da nije broj prosao ispod linije
    if(len(objects) == 0):
        M = cv2.moments(process_contours[0])
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        objects[ids+1] = (cX,cY)
        return
    # ako je duzina objects manja od broja kontura znaci da se javila nova i da je treba dodati
    # u suprotnom treba nastaviti sa obradom i osvezavanjem pozicija
    update_distances(process_contours)


def update_distances(process_contours):
    temp = dict()
    i = 0
    # dodaj nove polozaje centroida
    for contour in process_contours:
        M = cv2.moments(contour)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        temp[i+1] = (cX,cY)

    inputCentroids = np.array(temp.values())
    # argsort() - sortira od najmanjeg ka najvecem ali vraca indekse elemenata, ne vrednosti
    # argmin() - vraca indeks na kom se nalazi minimalni element
    for key, value in objects.items():
        if(inputCentroids.size != 0):
            idx = calculate_diff(value, inputCentroids)
            objects[key] = inputCentroids[idx]
            # obrisi iz liste posto je iskoriscen vec
            inputCentroids = np.delete(inputCentroids, idx)

    # ako nakon obrade ne bude prazan niz to znaci da je u medjuvremenu upala nova kontura
    # uzeti novu i dodati je sa novim id
    if(inputCentroids.size > 0):
        objects[ids+1] = inputCentroids[0]

    return

# funkcija uzima vrednosti iz liste i racuna razlike izmedju vrednosti i novih pozicija
# vraca indeks elementa koji predstavlja novu poziciju
def calculate_diff(value, inputCentroids):
    i = 0
    pom = np.array(inputCentroids)
    for i in range(len(inputCentroids)):
        distance = dist.cdist(value, inputCentroids[i])
        pom = np.append(pom, distance)

    index = pom.argmin()
    return index



# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
ids = 0;
objects = dict();

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')

cap = cv2.VideoCapture('resources/video-2.avi')
ret_begin, frame_begin = cap.read()

blurrr = cv2.GaussianBlur(frame_begin,(5,5),0)
blue_mask = blueLine(blurrr)
green_mask = greenLine(blurrr)

#HOUGH
blue_points = findLines(blue_mask,frame_begin)
green_points = findLines(green_mask,frame_begin)
alphabet = [0,1,2,3,4,5,6,7,8,9]
i = 0
while cap.isOpened():
    i += 1
    print("FRAME:", i)
    ret, frame = cap.read()
    ret_temp, frame_temp = cap.read()
    if ret != True:
        break
    if i == 350:
        break
    rects = []
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    img = image_bin(image_gray(frame))
    img_bin = erode(dilate(img))
    selected_regions, numbers, contours = select_roi(frame.copy(), img, rects)
    tst_ann = prepare_for_ann(numbers)
    result_ann = loaded_model.predict(np.array(tst_ann, np.float32))
    alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #print(display_result(result_cnn,alphabet))
    #print(display_result(result_ann, alphabet))


    tracker(contours, frame,ids,objects)
    cv2.imshow("Frame", frame)
    cv2.waitKey(50)
    #process(contours,frame,numbers,result_ann,alphabet)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

