from keras.models import model_from_json
from centroidtracker import CentroidTracker
from helpers import *
import math


def findLines(mask,frame):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(mask, 100, 300, apertureSize=5)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,maxLineGap=15,minLineLength=15)
    points = []
    miny1 = dict()
    miny2 = dict()
    # pronadji donji deo linija
    for line in lines:
        for x1, y1, x2, y2 in line:
            miny1[y1] = x1
            miny2[y2] = x2
    temp1 = list(miny1.keys())
    temp2 = list(miny2.keys())
    key1 = max(temp1)
    key2 = max(temp2)
    points.append(miny1[key1])
    points.append(key1)
    points.append(miny2[key2])
    points.append(key2)
    return points

# racuna daljinu konture od linije
def calculate_distance(x1, y1, x2, y2, x0, y0):
    denominator = (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1

    if denominator > 0:
        denominator = denominator * (-1)
    else:
        denominator = denominator * (-1)

    numerator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return denominator / numerator

def process_blue(contours, frame):
    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = cv2.boundingRect(contour)
        box = (x, y, x+w, y+h)
        X = int((x + x + w) / 2.0)
        Y = int((y + y + h) / 2.0)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if Y < blue_points[1] + 5 and X > blue_points[0] - 5 and X < blue_points[2] + 5:
            slope = calculate_slope(blue_points[0],blue_points[1],blue_points[2],blue_points[3])
            offset = calculate_offset(slope,blue_points[0],blue_points[1])
            if isUnderLine(x,y,slope,offset):
                rects.append(box)

    objects_blue = ct_blue.update(rects,blue_points[0],blue_points[1],blue_points[2],blue_points[3], frame, loaded_model)


def process_green(contours, frame):
    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = cv2.boundingRect(contour)
        box = (x, y, x + w, y + h)
        X = int((x + x + w) / 2.0)
        Y = int((y + y + h) / 2.0)
        if Y < green_points[1] + 5 and X > green_points[0] - 5 and X < green_points[2] + 5:
            # nadji jednacinu prave i proveri da li je ispod linije
            slope = calculate_slope(green_points[0], green_points[1], green_points[2], green_points[3])
            offset = calculate_offset(slope, green_points[0], green_points[1])
            if isUnderLine(x, y, slope, offset):
                rects.append(box)

    objects_green = ct_green.update(rects, green_points[0], green_points[1], green_points[2], green_points[3], frame, loaded_model)


# model = create_ann()
# model = train_ann(model)

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')

for i in range(10):
    # inicijalazicja centroidtrackera, posebno se koristi za zelenu i plavu liniju
    ct_blue = CentroidTracker()
    ct_blue.setType("BLUE")
    ct_green = CentroidTracker()
    ct_green.setType("GREEN")

    cap = cv2.VideoCapture('resources/video-' + str(i) + '.avi')
    ret_begin, frame_begin = cap.read()
    # maske za pronalazenje linija
    blurrr = cv2.GaussianBlur(frame_begin,(5,5),0)
    blue_mask = blueLine(blurrr)
    green_mask = greenLine(blurrr)

    #HOUGH
    blue_points = findLines(blue_mask,frame_begin)
    green_points = findLines(green_mask,frame_begin)
    alphabet = [0,1,2,3,4,5,6,7,8,9]
    while cap.isOpened():
        ret, frame = cap.read()
        ret_temp, frame_temp = cap.read()
        if ret != True:
            break

        img = image_bin(image_gray(frame))
        selected_regions, numbers, contours = select_roi(frame.copy(), img)
        process_blue(contours, frame)
        process_green(contours, frame)
        cv2.imshow("FRAME" + str(i), frame)
        cv2.waitKey(3)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    f = open('resources/out.txt','a')
    f.write("video-"+ str(i) +".avi " + str(ct_blue.getSum() - ct_green.getSum()) + "\n")
    f.close()

    cap.release()
    cv2.destroyAllWindows()

