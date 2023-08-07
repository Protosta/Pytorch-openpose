import numpy as np
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

neck_length = []
body_length = []
shoulder_width = []
leg_length = []
crotch_length = []
neck_angle = []
arm_length = []

def rad_to_deg(rad):
    return rad * (180 / math.pi)

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):

    pos = []
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            pos_line =[]
            pos_line.append(int(x))
            pos_line.append(int(y))
            pos.append(pos_line)
            print(pos)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


def body_training(candidate, subset):
    pos = []
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            pos_line =[]
            pos_line.append(int(x))
            pos_line.append(int(y))
            pos.append(pos_line)
            #print(pos[i][0])
           # print(pos[i][1])
    neck = pos[1][1]-pos[0][1]
    body = pos[10][1] - pos[1][1]
    shoulder = pos[5][0]-pos[2][0]
    crotch = pos[11][0]-pos[8][0]
    global neck_length,body_length,shoulder_width,crotch_length,arm_length
    neck_length =np.append(neck_length, format(neck,'.1f' ))
    body_length = np.append(body_length, format(body, '.1f'))
    shoulder_width = np.append(shoulder_width, format(shoulder, '.1f'))
    crotch_length = np.append(crotch_length, format(crotch, '.1f'))

def body_side_training(candidate, subset):
    pos = []
    pos_line = []
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                pos_line.append(0)
                pos_line.append(0)
                #continue
            x, y = candidate[index][0:2]
            pos_line.append(int(x))
            pos_line.append(int(y))
            pos.append(pos_line)
    arm = pos[6][1] - pos[1][1]
    leg = pos[10][1]-pos[8][1]
    c_2 = (pos[0][1]-pos[8][1])**2+(pos[0][0]-pos[8][0])**2
    a_2 = (pos[0][1]-pos[1][1])**2+(pos[0][0]-pos[1][0])**2
    b_2 = (pos[1][1]-pos[8][1])**2+(pos[1][0]-pos[8][0])**2
    neck = rad_to_deg(math.acos((a_2 + b_2 - c_2) / (2 * math.sqrt(a_2 * b_2))))
    global neck_angle,arm_length,leg_length
    neck_angle = np.append(neck_angle, format(neck, '.1f'))
    arm_length = np.append(arm_length, format(arm, '.1f'))
    leg_length = np.append(leg_length, format(leg, '.1f'))
def body_test(candidate, subset):
    neck = candidate[1][1]-candidate[0][1]
    body = candidate[10][1] - candidate[1][1]
    shoulder = candidate[5][0]-candidate[2][0]
    leg = candidate[10][1]-candidate[8][1]
    global neck_length,body_length,shoulder_width,leg_length
    variance_min = 10000
    variance_number = -1
    body_length_float = np.asarray(body_length, dtype=float)
    neck_length_float = np.asarray(neck_length, dtype=float)
    leg_length_float = np.asarray(leg_length, dtype=float)
    shoulder_width_float = np.asarray(shoulder_width, dtype=float)

    for i in range(len(neck_length)):
        variance = abs(body - body_length_float[i]) + abs(shoulder - shoulder_width_float[i]) + abs(leg - leg_length_float[i]) + abs(neck - neck_length_float[i])
        if (variance_min > variance):
            variance_number = i
            variance_min = variance
    print(variance_min)
    print(variance_number)


def print_neck_length():
    return neck_length
def print_body_length():
    return body_length
def print_shoulder_width():
    return shoulder_width
def print_leg_length():
    return leg_length
def print_crotch_length():
    return crotch_length
def print_nack_angle():
    return neck_angle
def print_arm_length():
    return arm_length
# get max index of 2d area
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
