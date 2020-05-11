# Active Query Imager Testbench
# Author: Rajkumar Kubendran

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import struct
import time
import ok
import dac_control as dac
import os
import csv
import random as rn
from scipy.sparse.csc import csc_matrix
import more_itertools as mit
#dl libraraies
from keras import backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
# import tkinter as tk
from scipy.ndimage import convolve
record = False
predict_gesture = False
predict_direction = False
threshold = 60
test_data = False
prep_data = False
background_filter = False

def sigmoid (x, A, h, slope, offset):
    return 1 / (1 + np.exp((x - h) / slope)) * A + offset

def get_sigmoid_params(x, y):
    amplitude = np.max(y) - np.min(y)
    if amplitude > 0.5:
        # print(x.shape)
        offset = x[np.minimum(np.apply_along_axis(lambda a: a.searchsorted(0.5*amplitude+np.min(y)), axis = 0, arr = y), np.size(x)-1)]
        slope = x[np.minimum(np.apply_along_axis(lambda a: a.searchsorted(1/(1+1/np.exp(1))+np.min(y)), axis = 0, arr = y), np.size(x)-1)] - offset
    else:
        offset = np.mean(x)
        slope = 10000 #np.log(0)
    return slope, offset

def offset_calib(img_out_pos, slope_pos, offset_pos, img_out_neg, slope_neg, offset_neg):
    img_inp_pos = np.zeros((65536, ))
    img_calib_pos = np.zeros((256, 256))
    img_inp_neg = np.zeros((65536,))
    img_calib_neg = np.zeros((256, 256))
    # print(np.min(img_out_pos), np.max(img_out_pos))
    # print(np.min(img_out_neg), np.max(img_out_neg))
    img_inp_pos = slope_pos * np.log(np.reshape(img_out_pos / (1 - img_out_pos), (65536,))) + offset_pos
    # print(np.min(img_inp_pos), np.max(img_inp_pos))
    img_calib_pos = np.reshape(sigmoid(img_inp_pos + offset_pos, -1, offset_pos, slope_pos, 1), (256, 256))
    # print(np.min(img_calib_pos), np.max(img_calib_pos))
    img_inp_neg = slope_neg * np.log(np.reshape(img_out_neg / (1 - img_out_neg), (65536,))) + offset_neg
    # print(np.min(img_inp_neg), np.max(img_inp_neg))
    img_calib_neg = np.reshape(sigmoid(img_inp_neg + offset_neg, -1, offset_neg, slope_neg, 1), (256, 256))
    # print(np.min(img_calib_neg), np.max(img_calib_neg))
    return (img_calib_pos + img_calib_neg)/2

def click_and_record(event, x, y, flags, param):
    global record
    if event == cv2.EVENT_LBUTTONDOWN:
        # print("Mouse click detected...")
        record = True
    else:
        record = False

def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) > 5:
            yield group[0], group[-1]

blurValue = 41


def get_contour(img):
    ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length): # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        res = contours[ci]
#         hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (255, 0, 0), 2)
    #     cv2.drawContours(drawing, [hull], 0, (255, 0, 0), 1)
    return drawing


if predict_gesture:
    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir('gesture_training/'):
        if not j.startswith('.'): # If running this code locally, this is to
            # ensure you aren't reading in hidden folders
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1
    print(lookup)

    model = load_model('cnn_qdvs_gesture.h5')
    # model.summary()
    # model.evaluate(x_test, y_test)

    if test_data:
        if prep_data:
            x_data = []
            y_data = []
            IMG_SIZE = 256
            datacount = 0 # We'll use this to tally how many images are in our dataset
            for i in range(0, 10): # Loop over the ten top-level folders
                for j in range(0, 10): #os.listdir('leapgestrecog/leapGestRecog/0' + str(i) + '/'):np.array([0, 1, 2, 6])
                    # if not j.startswith('.'): # Again avoid hidden folders
                    count = 0 # To tally images of a given gesture
                    for k in os.listdir('leapgestrecog/leapGestRecog/0' + str(i) + '/' + reverselookup[j] + '/'):
                        # Loop over the images
                        path = 'leapgestrecog/leapGestRecog/0' + str(i) + '/' + reverselookup[j] + '/' + k
                        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                        arr = np.array(get_contour(img))
                        x_data.append(arr)
                        count = count + 1
                    y_values = np.full((count, 1), j)
                    y_data.append(y_values)
                    datacount = datacount + count
            x_data = np.array(x_data, dtype = 'float32')
            y_data = np.array(y_data)
            y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size
            y_data = to_categorical(y_data)
            x_data = x_data.reshape((datacount, IMG_SIZE, IMG_SIZE, 1))/255
            # x_data = x_data/255
            x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.25,random_state=42)

            np.save("x_test", x_test)
            np.save("y_test", y_test)
            np.save("x_train", x_train)
            np.save("y_train", y_train)
        else:
            x_test = np.load("x_test.npy")
            y_test = np.load("y_test.npy")

        # prepare input image
        image_index = 1
        # plt.imshow(x_test[image_index].reshape(256, 256),cmap='Greys')
        # plt.show()
        for image_index in range(100):
            image = x_test[image_index].reshape(1, 256, 256, 1)

            probs = model.predict(image)
            prediction = reverselookup[probs.argmax()][3:]
            score = np.round(100*probs[0, probs.argmax()], 2)
            # print(prediction, score)
            # print(reverselookup[y_test[image_index].argmax()])

            cv2.putText(x_test[image_index], f"Prediction: {prediction} ({score}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255))
            # cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,255, 255, 255))
            cv2.imshow('check', x_test[image_index])
            cv2.waitKey(500)



# Initialize Opal Kelly
dev = ok.okCFrontPanel()

# Load FPGA program
dev.OpenBySerial("")
dev.ConfigureFPGA("C:/Users/rchin/PycharmProjects/qDVS/qdvs_latest.bit")

eventmode = False

# Reset chip
dev.SetWireInValue(0x00, 0b1)
dev.UpdateWireIns()
# Set Mode of operation
if eventmode:
    dev.SetWireInValue(0x00, 0x080) #520 MUST set eventmode flag correctly!
else:
    dev.SetWireInValue(0x00, 0x280)
dev.UpdateWireIns()
# Set clock frequency
if eventmode:
    dev.SetWireInValue(0x12, 0xFF000+0b11) #0xCF7F0
else:
    dev.SetWireInValue(0x12, 0b11)
dev.UpdateWireIns()
# Set Frame reset delay
dev.SetWireInValue(0x13, 0x00000)
dev.UpdateWireIns()

# Program all DACs
dac.dac_program_all(dev)
if eventmode:
    dac.vdac_program_single_daisy(dev, 0, 0, 0.2)  # VDAC_EXT_DAC 0.2
    dac.vdac_program_single_daisy(dev, 0, 1, 0.35)  # VUP 0.35 25
    dac.vdac_program_single_daisy(dev, 0, 2, 0.25)  # VDN 0.25 15
    dac.vdac_program_single_daisy(dev, 0, 3, 0.2)  # VREF/VMID 0.2
    dac.vdac_program_single_daisy(dev, 1, 2, 0.4)  # VOUT_RST_DC
    dac.vdac_program_single_daisy(dev, 1, 3, 0.7)  # VCM
    dac.vdac_program_single_daisy(dev, 1, 6, 0.0)  # NBIAS_OFF
    dac.vdac_program_single_daisy(dev, 1, 7, 0.0)  # NBIAS_ON
else:
    dac.vdac_program_single_daisy(dev, 0, 0, 0.4)  # VDAC_EXT_DAC 0.2
    dac.vdac_program_single_daisy(dev, 0, 1, 0.35)  # VUP 0.35 25
    dac.vdac_program_single_daisy(dev, 0, 2, 0.25)  # VDN 0.25 15
    dac.vdac_program_single_daisy(dev, 0, 3, 0.4)  # VREF/VMID 0.2
    dac.vdac_program_single_daisy(dev, 1, 2, 0.4)  # VOUT_RST_DC
    dac.vdac_program_single_daisy(dev, 1, 3, 0.6)  # VCM
    dac.vdac_program_single_daisy(dev, 1, 6, 0.5)  # NBIAS_OFF
    dac.vdac_program_single_daisy(dev, 1, 7, 0.0)  # NBIAS_ON

# Mode of operation
sweep = False
display = True


dt = 0.02
decay_time = 0.1
decay_scale = np.exp(-dt/decay_time)
aq_image = np.zeros((256, 256))

buf_size = 4*256*256 # Bytes in 1 Frame = 256*256 pixels x 4bytes/pixel = Same as FIFO Size in FPGA
now = None
fps = 30
frame_cnt = 0
record_cnt = 0
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, (256, 256))

# Threshold Voltage sweep
sweep_max = 0.4
sweep_min = 0.2
sweep_step = 0.01
sweep_pts = np.uint8((sweep_max-sweep_min)/sweep_step)+1
vsweep = np.array([i for i in range(sweep_pts)])

pxl_cnt = np.ones((sweep_pts, 65536))
pxl_avg = np.zeros((sweep_pts, 65536))
# pxl_var = np.zeros((sweep_pts, 65536))
frame_avg = np.zeros((sweep_pts, ))
frame_std = np.zeros((sweep_pts, ))

if sweep:
    dac.vdac_program_single_daisy(dev, 0, 1, sweep_max)  # VUP
    dac.vdac_program_single_daisy(dev, 0, 2, sweep_min)  # VDN
    # Sweep VUP
    for i in range(sweep_pts):
        dac.vdac_program_single_daisy(dev, 0, 1, i*sweep_step + sweep_min)
        frame_cnt = 0
        while frame_cnt < 100:
            buf = bytearray(buf_size)
            dev.ReadFromPipeOut(0xa3, buf)
            end = buf_size - 1
            buf = np.array(list(buf))

            x = buf[0:end:4]
            y = buf[1:end:4]
            p = buf[2:end:4]
            # p = np.sign(np.int8(p << 6))
            np.add.at(pxl_cnt[i, :], [y + 256 * x], 1)
            np.add.at(pxl_avg[i, :], [y + 256 * x], p)
            # np.add.at(pxl_var[i, :], [y + 256 * x], p**2)
            frame_cnt += 1
        # pxl_avg[i] = avg_img[128, 128]

    pxl_avg = -0.5*(pxl_avg/pxl_cnt)
    frame_avg = np.squeeze(np.mean(pxl_avg, axis=1))
    frame_std = np.squeeze(np.std(pxl_avg, axis=1))

    filtered = []
    pxl_avg_tran = pxl_avg.T
    for y in pxl_avg_tran.tolist():
        check = np.all(np.array(y) - frame_avg + 3 * frame_std > 0) and np.all(np.array(y) - frame_avg - 3 * frame_std < 0)
        if check:
            filtered.append(y)
    final_list = np.array(filtered)
    final_list = final_list.T
    print(final_list.shape)
    plt.figure(1)
    plt.plot([i * sweep_step + sweep_min for i in range(sweep_pts)], final_list[:, [x * 257 for x in range(128)]])
    # plt.fill_between([i * 0.05 for i in range(10)], pxl_avg[:, [x * 257 for x in range(256)]]-pxl_var[:, [x * 257 for x in range(256)]], pxl_avg[:, [x * 257 for x in range(256)]]+pxl_var[:, [x * 257 for x in range(256)]])
    plt.xlabel('VUP (V)')
    plt.ylabel('Frequency of Events')
    plt.title('Pixel Characterization')
    plt.grid(True)

    frame_flt_avg = np.squeeze(np.mean(final_list, axis=1))
    frame_flt_std = np.squeeze(np.std(final_list, axis=1))
    plt.figure(2)
    # params, _ = curve_fit(sigmoid, vsweep, frame_flt_avg)
    slope_pos, offset_pos = get_sigmoid_params(vsweep, final_list + np.ones(final_list.shape))# pxl_avg + np.ones(pxl_avg.shape)
    # np.save("slope_pos", slope_pos)
    # np.save("offset_pos", offset_pos)
    # Offset Compensation
    # plt.plot([(i - offset_pos[[x * 257 for x in range(128)]]) * sweep_step + sweep_min for i in range(sweep_pts)], pxl_avg[:, [x * 257 for x in range(128)]])
    # Filtered frame - not good
    # plt.plot([i * sweep_step for i in range(sweep_pts)], frame_flt_avg, 'k--', label='original')
    # Filtered pixels
    plt.plot([i * sweep_step + sweep_min for i in range(sweep_pts)], sigmoid(vsweep, -1, np.mean(offset_pos), np.mean(slope_pos), 0), label='sigmoid fit')
    # plt.fill_between([i*sweep_step for i in range(sweep_pts)], np.maximum(-1*np.ones((sweep_pts, )), frame_flt_avg-frame_flt_std), np.minimum(np.zeros((sweep_pts, )), frame_flt_avg+frame_flt_std), alpha=0.5)
    plt.fill_between([i * sweep_step + sweep_min for i in range(sweep_pts)],
                     np.maximum(-1 * np.ones((sweep_pts,)), sigmoid(vsweep, -1, np.mean(offset_pos) - 3*np.std(offset_pos), np.mean(slope_pos), 0)),
                     np.minimum(np.zeros((sweep_pts,)), sigmoid(vsweep, -1, np.mean(offset_pos) + 3*np.std(offset_pos), np.mean(slope_pos), 0)), alpha=0.5)
    plt.legend()
    plt.xlabel('VUP (V)')
    plt.ylabel('Frequency of Events')
    plt.title('Pixel Characterization')
    plt.text(0.6, -0.8, 'Slope = %s' %np.around(np.mean(slope_pos), decimals=10))
    plt.text(0.6, -0.9, 'Offset = %s mV' %np.around(np.mean(offset_pos)*sweep_step*1000, decimals=10))
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.grid(True)
    # plt.show()
    # print(params)
    print("Slope = ", 1/np.around(np.mean(slope_pos), decimals=10))
    print("Mean Offset (mV) = ", np.around(np.mean(offset_pos) * sweep_step * 1000, decimals=10))
    print("Offset spread (mV) = ", np.around(6*np.std(offset_pos) * sweep_step * 1000, decimals=10))
    frame_avg_combined = frame_flt_avg

    frame_cnt = 0
    pxl_cnt = np.ones((sweep_pts, 65536))
    pxl_avg = np.zeros((sweep_pts, 65536))
    pxl_var = np.zeros((sweep_pts, 65536))
    frame_avg = np.zeros((sweep_pts,))
    frame_var = np.zeros((sweep_pts,))
    # Sweep VDN
    for i in range(sweep_pts):
        dac.vdac_program_single_daisy(dev, 0, 2, i * sweep_step + sweep_min)
        frame_cnt = 0
        while frame_cnt < 100:
            buf = bytearray(buf_size)
            dev.ReadFromPipeOut(0xa3, buf)
            end = buf_size - 1
            buf = np.array(list(buf))

            x = buf[0:end:4]
            y = buf[1:end:4]
            p = buf[2:end:4]
            # p = np.sign(np.int8(p << 6))
            np.add.at(pxl_cnt[i, :], [y + 256 * x], 1)
            np.add.at(pxl_avg[i, :], [y + 256 * x], p)
            # np.add.at(pxl_var[i, :], [y + 256 * x], p**2)
            frame_cnt += 1


    pxl_avg = pxl_avg / pxl_cnt
    frame_avg = np.squeeze(np.mean(pxl_avg, axis=1))
    frame_std = np.squeeze(np.std(pxl_avg, axis=1))

    filtered = []
    pxl_avg_tran = pxl_avg.T
    for y in pxl_avg_tran.tolist():
        check = np.all(np.array(y) - frame_avg + 3 * frame_std > 0) and np.all(
            np.array(y) - frame_avg - 3 * frame_std < 0)
        if check:
            filtered.append(y)
    final_list = np.array(filtered)
    final_list = final_list.T
    print(final_list.shape)

    plt.figure(3)
    plt.plot([i * sweep_step + sweep_min for i in range(sweep_pts)], final_list[:, [x * 257 for x in range(128)]])
    # plt.fill_between([i * 0.05 for i in range(10)], pxl_avg[:, [x * 257 for x in range(256)]]-pxl_var[:, [x * 257 for x in range(256)]], pxl_avg[:, [x * 257 for x in range(256)]]+pxl_var[:, [x * 257 for x in range(256)]])
    plt.xlabel('VDN (V)')
    plt.ylabel('Frequency of Events')
    plt.title('Pixel Characterization')
    plt.grid(True)

    frame_flt_avg = np.squeeze(np.mean(final_list, axis=1))
    frame_flt_std = np.squeeze(np.std(final_list, axis=1))
    plt.figure(4)
    # params, _ = curve_fit(sigmoid, vsweep, frame_flt_avg)
    slope_neg, offset_neg = get_sigmoid_params(vsweep, final_list) #
    # np.save("slope_neg", slope_neg)
    # np.save("offset_neg", offset_neg)
    # plt.plot([i * sweep_step for i in range(sweep_pts)], frame_flt_avg, 'k--', label='original')
    # plt.plot([i * sweep_step for i in range(sweep_pts)], sigmoid(vsweep, *params), label='sigmoid fit')
    # Offset Compensation
    # plt.plot([(i - offset_neg[[x * 257 for x in range(128)]]) * sweep_step + sweep_min for i in range(sweep_pts)], pxl_avg[:, [x * 257 for x in range(128)]])
    # Filtered pixels
    plt.plot([i * sweep_step + sweep_min for i in range(sweep_pts)], sigmoid(vsweep, -1, np.mean(offset_neg), np.mean(slope_neg), 1),
             label='sigmoid fit')
    plt.fill_between([i * sweep_step + sweep_min for i in range(sweep_pts)],
                     np.maximum(np.zeros((sweep_pts,)), sigmoid(vsweep, -1, np.mean(offset_neg) - 3*np.std(offset_neg), np.mean(slope_neg), 1)),
                     np.minimum(np.ones((sweep_pts,)), sigmoid(vsweep, -1, np.mean(offset_neg) + 3*np.std(offset_neg), np.mean(slope_neg), 1)), alpha=0.5)
    # plt.fill_between([i * sweep_step for i in range(sweep_pts)],
    #                  np.maximum(np.zeros((sweep_pts,)), frame_flt_avg - frame_flt_std),
    #                  np.minimum(np.ones((sweep_pts,)), frame_flt_avg + frame_flt_std), alpha=0.5)
    plt.legend()
    plt.xlabel('VDN (V)')
    plt.ylabel('Frequency of Events')
    plt.title('Pixel Characterization')
    plt.text(0.6, 0.2, 'Slope = %s' % np.around(np.mean(slope_neg), decimals=10))
    plt.text(0.6, 0.1, 'Offset = %s mV' % np.around(np.mean(offset_neg)*sweep_step*1000, decimals=10))
    plt.grid(True)
    # plt.show()
    # print(params)
    print("Slope = ", 1/np.around(np.mean(slope_neg), decimals=10))
    print("Mean Offset (mV) = ", np.around(np.mean(offset_neg)*sweep_step*1000, decimals=10))
    print("Offset spread (mV) = ", np.around(6 * np.std(offset_neg) * sweep_step * 1000, decimals=10))

    # frame_avg_combined = np.append(frame_avg_combined, frame_flt_avg)
    # plt.figure(5)
    # plt.plot([(i * sweep_step)- (sweep_max-sweep_min) for i in range(2*sweep_pts)], frame_avg_combined)
    plt.show()

prev_frame = np.zeros((5, 256, 256))
delta_frame = np.zeros((256, 256))
# dac.vdac_program_single_daisy(dev, 0, 1, 0.29)  # VUP 0.45
# dac.vdac_program_single_daisy(dev, 0, 2, 0.25)  # VDN 0.25
# slope_pos = np.load("slope_pos.npy")
# print(slope_pos.shape)
# print(np.min(slope_pos), np.max(slope_pos))
# slope_neg = -1*np.load("slope_neg.npy")
# print(np.min(slope_neg), np.max(slope_neg))
# offset_pos = np.load("offset_pos.npy")
# print(np.min(offset_pos), np.max(offset_pos))
# offset_neg = np.load("offset_neg.npy")
# print(np.min(offset_neg), np.max(offset_neg))
n_white = 0
n_black = 0
white_max = 0
white_min = 65536
black_max = 0
black_min = 65536
aq_image = np.zeros((256, 256))#+0.5
testimg = np.zeros((256, 256))
dispimg = np.zeros((256, 256))
buf = bytearray(buf_size)
end = buf_size - 1
buf = np.array(list(buf))
p_prev = buf[2:end:4]
if predict_gesture:
    prediction = reverselookup[1]#[3:]
    score = 0
# print(np.size(p_prev))
direction = ''

timestamp_map = np.zeros((256, 256))
spatial_correlation_map = np.zeros((256, 256))
tstamp = 0
tstamp_prev = 0
tstop = 0
tstart = 0
tperiod = 0
cv2.namedWindow("qDVS Output")
cv2.setMouseCallback("qDVS Output", click_and_record)
# fgbg = cv2.createBackgroundSubtractorKNN()
tick_elapse = 0
tick_start = time.time()
kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

while display: #tick_elapse < 10
    try:
        tstart = time.time()
        if eventmode: # Comment out for exponential decay
            aq_image = np.zeros((256, 256))+0.5
        # else:
        #     aq_image = np.zeros((256, 256))
        img_pos = np.zeros((256, 256))
        img_neg = np.zeros((256, 256))
        end = buf_size - 1
        n_white = 0
        n_black = 0
        direction = ''

        for read_cnt in range(1):
            buf = bytearray(buf_size)
            dev.ReadFromPipeOut(0xa3, buf)
            # aq_image *= decay_scale # Remove comment for exponential decay

            buf = np.array(list(buf))

            y = buf[0:end:4] #[x for x in buf[0:end:4]]
            x = buf[1:end:4]
            p = buf[2:end:4]

            if eventmode:
                n_white = np.count_nonzero(p == 2)
                n_black = np.count_nonzero(p == 1)
                p = -np.sign(np.int8(p << 6)) / 2 #+ 0.5 Remove /2 for exponential decay
                # p_prev = p + p_prev*np.exp(-dt / decay_time)

                # print(x[0:50], y[0:50], p[0:50])
                # Convert -1, 0, 1 events into black, gray and white pixels
                activity = np.round((n_black + n_white) * 100 / 65536, 2)
                event_matrix = csc_matrix((p, (x, y)), shape=(256, 256))
                rows, cols = event_matrix.nonzero()
                # print(event_matrix[rows, cols])
                # if np.remainder(frame_cnt, 10) == 0:
                #     print(np.maximum(rows, 10), np.min(rows), np.max(cols), np.min(cols))
            else:
                # print(int(np.average(p)), int(np.std(p)))
                print(np.max(p), np.min(p))
                # print("rows", x[120:130], "cols", y[120:130], "vals", p[120:130])
                p = p/63#1 - p/31 #1 - p/15 #np.int8(~p << 2)/127
                # event_matrix = csc_matrix((p, (x, y)), shape=(256, 256))
                # rows, cols = event_matrix.nonzero()


            if background_filter:
                tstamp = 1000 * tstart + tperiod * (1 - rows / 256) + tperiod * (1 - cols / 256) / 256
                tstamp_prev = timestamp_map[rows, cols]
                timestamp_map[[np.maximum(rows - 1, 0), rows, np.minimum(rows + 1, 255)], [
                    np.maximum(cols - 1, 0), cols, np.minimum(cols + 1, 255)]] = tstamp
                # print(np.round(tstamp-tstamp_prev), 2)
                # event_matrix[rows, cols] = np.multiply(event_matrix[rows, cols], ((tstamp - tstamp_prev) > 100))#1.5 * tperiod))
                # timestamp_map[rows, cols] = np.multiply(tstamp_prev, ((tstamp - tstamp_prev) < 100))
                # print(event_matrix.shape, kernel.shape)
                aq_image[rows, cols] = aq_image[rows, cols] + event_matrix[rows, cols]/2
                print(np.max(delta_frame), np.min(delta_frame))
                spatial_correlation_map = np.multiply(2*aq_image-1, convolve(2*aq_image-1, kernel))
                # print(np.max(spatial_correlation_map), np.min(spatial_correlation_map))
                # aq_image = np.multiply(aq_image-0.5, spatial_correlation_map > 1) + 0.5
                aq_image = np.multiply(aq_image - 0.5, delta_frame > 0) + 0.5
                # print(np.max(aq_image), np.min(aq_image))
                # for pxl_event in range(65536):
                #     if p[pxl_event] != 0:
                #         tstamp = 1000*tstart + tperiod*(1-x[pxl_event]/256) + tperiod*(1-y[pxl_event]/256)/256 #time.time() #
                #         # print("curernt time stamp =", tstamp)
                #         tstamp_prev = timestamp_map[x[pxl_event], y[pxl_event]]
                #         # print("previous time stamp =", tstamp_prev)
                #         # print("delta time stamp =", tstamp - tstamp_prev)
                #         # x_neighbors = [np.maximum(x[pxl_event] - 1, 0), x[pxl_event], np.minimum(x[pxl_event] + 1, 255)]
                #         # y_neighbors = [np.maximum(y[pxl_event] - 1, 0), y[pxl_event], np.minimum(y[pxl_event] + 1, 255)]
                #         # print("X neighbors = ", x_neighbors)
                #         # print("Y neighbors = ", y_neighbors)
                #         # print("Neigborhood tstamp before update =", timestamp_map[x_neighbors, y_neighbors])
                #         timestamp_map[[np.maximum(x[pxl_event] - 1, 0), x[pxl_event], np.minimum(x[pxl_event] + 1, 255)], [np.maximum(y[pxl_event] - 1, 0), y[pxl_event], np.minimum(y[pxl_event] + 1, 255)]] = tstamp
                #         # print("Neigborhood tstamp after update =", timestamp_map[x_neighbors, y_neighbors])
                #         if (tstamp - tstamp_prev) > 10*tperiod:
                #             p[pxl_event] = 0
                #             # timestamp_map[x[pxl_event], y[pxl_event]] = tstamp_prev

            # np.add.at(aq_image, [x, y], p) #+0.5 *decay_scale for exponential decay
            if eventmode:
                aq_image[rows, cols] = aq_image[rows, cols] + event_matrix[rows, cols]
            else:
                # np.add.at(aq_image, [x, y], p)
                aq_image[x,y] = p #0.2*p + 0.8*aq_image[x,y]
                # aq_image[rows, cols] = event_matrix[rows, cols]
                # print(np.max(aq_image), np.min(aq_image))
                # print(aq_image[120:130,120:130]*63)
            # print(rows[0:50], cols[0:50], aq_image[rows, cols])
            frame_cnt += 1
            if predict_direction:
                obj_rows = list(find_ranges(np.unique(rows)))
                obj_cols = list(find_ranges(np.unique(cols)))
                if obj_rows == [] or obj_cols == []:
                    direction = 'none'
                else:
                    if np.sum(event_matrix[np.max(obj_rows), :]) < 0:
                        direction = 'down'
                    if np.sum(event_matrix[np.min(obj_rows), :]) < 0: #np.min(rows):np.min(rows)+5
                        direction = 'up'
                    if np.sum(event_matrix[:, np.max(obj_cols)]) < 0: #np.max(cols)-5:np.max(cols)
                        direction += ' right'
                    if np.sum(event_matrix[:, np.min(obj_cols)]) < 0:
                        direction += ' left'



        # now += dt
        if frame_cnt > 10:
            # plt.ion()
            # plt.subplot(2, 1, 1)
            # plt.ylabel('Percentage of white pixels')
            # plt.title('Pixel Characterization')
            # plt.plot(frame_cnt, (100*n_white)/65536, color='r', marker='o')
            # plt.subplot(2, 1, 2)
            # plt.plot(frame_cnt, (100*n_black)/65536, color='b', marker='o')
            # plt.xlabel('Frame Number')
            # plt.ylabel('Percentage of black pixels')
            #
            # plt.pause(0.00001)
            # plt.show()
            # resize image
            if eventmode:
                if (n_white > white_max):
                    white_max = n_white
                    # print("Max % of WHITE pixels =", (100*white_max)/65536)
                # if (n_white < white_min):
                #     white_min = n_white
                #     print("Min % of WHITE pixels =", (100 * white_min) / 65536)
                if (n_black > black_max):
                    black_max = n_black
                    # print("Max % of BLACK pixels =", (100*black_max)/65536)
                # if(n_black < black_min):
                #     black_min = n_black
                #     print("Min % of BLACK pixels =", (100 * black_min) / 65536)

            # print(np.max(aq_image), np.min(aq_image))
            # aq_image_thresh = aq_image
            # pos_events = aq_image_thresh > 0.6
            # aq_image_thresh[pos_events] = 1.0
            # neg_events = aq_image_thresh < 0.4
            # aq_image_thresh[neg_events] = 0.0
            # fgmask = fgbg.apply(np.uint8(255 * aq_image))
            resized = cv2.resize(aq_image, (512, 512), interpolation=cv2.INTER_CUBIC) # aq_image/2+0.5 for exponential decay
            if predict_gesture:
                if np.remainder(frame_cnt, 5) == 0:
                    # testimg = testimg/10
                    probs = model.predict(aq_image.reshape(1, 256, 256, 1))
                    prediction = reverselookup[probs.argmax()]#[3:]
                    score = np.round(100 * probs[0, probs.argmax()], 2)
                    # print(prediction,score)
                    cv2.putText(aq_image, f"Prediction: {prediction} ({score}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0))
                    cv2.putText(resized, f"Prediction: {prediction} ({score}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0))
                    # testimg = np.zeros((256, 256))
                else:
                    cv2.putText(aq_image, f"Prediction: {prediction} ({score}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0))
                    cv2.putText(resized, f"Prediction: {prediction} ({score}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0))
                    # testimg += aq_image
            # if activity > 1: #0.1
            #     cv2.putText(resized, f"Activity: {activity}%", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                 (255, 0, 0))

            # finalimg = cv2.fastNlMeansDenoising(np.uint8(255 * resized), None, 20, 7, 21)
            # edges = cv2.Canny(np.uint8(255 * resized), 20, 30)
            # canny_images = np.hstack((resized, edges))
            # cv2.rectangle(aq_image, (rectx, recty), (rectx+rectw, recty+recth), (255,255,255), 2)
            if eventmode:
                # K Means Clustering
                # pixel_values = np.float32(resized.reshape((-1, 1)))
                # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                # k = 1 # K-Means Clustering
                # _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                # centers = np.uint8(centers)
                # print(centers)

                clr_image = cv2.cvtColor(np.uint8(255 * resized), cv2.COLOR_GRAY2RGB)
                color = (255, 0, 0)
                thickness = 2
                # print(list(find_ranges(np.unique(rows))), list(find_ranges(np.unique(cols))))
                obj_rows = list(find_ranges(np.unique(rows)))
                obj_cols = list(find_ranges(np.unique(cols)))
                if obj_rows == [] or obj_cols == []:
                    start_point = (0, 0)
                    end_point = (0, 0)
                    value = (255<<12)+(0<<4)+0b11 #0xFF000 + 0b11
                    # print(value)
                else:
                    for x in range(len(obj_rows)):
                        for y in range(len(obj_cols)):
                            if (np.sum(np.abs(event_matrix[obj_rows[x][0]:obj_rows[x][1], obj_cols[y][0]:obj_cols[y][1]]))) > 10:
                                start_point = (2*obj_cols[y][0], 2*obj_rows[x][0])  # (5, 5)
                                end_point = (2*obj_cols[y][1], 2*obj_rows[x][1])
                                # color = np.uint8(255*np.random.random(3)).tolist()
                                clr_image = cv2.rectangle(clr_image, start_point, end_point, color, thickness)
                    # start_point = (2*np.min(obj_cols), 2*np.min(obj_rows)) #(5, 5)
                    # end_point = (2*np.max(obj_cols), 2*np.max(obj_rows)) #(220, 220)
                    # value = (int(start_point[1])<<4)+(int(end_point[1])<<12)+0b11
                    # print(value)
                    # dev.SetWireInValue(0x12, ((start_point[1] << 4) + (end_point[1] << 12) + 0b11))
                # dev.SetWireInValue(0x12, value)
                # dev.UpdateWireIns()


                # Using cv2.rectangle() method
                # Draw a rectangle with blue line borders of thickness of 2 px
                # clr_image = cv2.rectangle(clr_image, start_point, end_point, color, thickness)
                if predict_direction:
                    cv2.putText(clr_image, f"Direction: {direction}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0))
                # event_pRGB = cv2.cvtColor(np.uint8(255 * aq_image), cv2.COLOR_GRAY2RGB)
                out.write(clr_image)
                cv2.imshow("qDVS Output", clr_image)
            else:
                # event_pRGB = cv2.cvtColor(np.uint8(255 * aq_image), cv2.COLOR_GRAY2RGB)
                # out.write(np.uint8(event_pRGB))
                cv2.imshow("qDVS Output", resized)
        if record:
            # if activity > 3:
                print("Recording frame...")
                event_pRGB = cv2.cvtColor(np.uint8(255 * aq_image), cv2.COLOR_GRAY2RGB)
                out.write(np.uint8(event_pRGB))
                # cv2.imwrite("gesture_training/stereo_left_"+str(record_cnt)+".png", event_pRGB)
                record_cnt += 1


        tstop = time.time()
        tick_elapse = tstop - tick_start
        tperiod = 1000 * (tstop - tstart)
        # print("Time elapsed in ms = ", tperiod)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # plt.pause(10)
            plt.ioff()
            # plt.show()
            dev.__del__
            break

    except KeyboardInterrupt:
        dev.__del__
        out.release()
        break

# with open(r'waterfall.csv', 'a', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow([(100*white_max)/65536, (100*black_max)/65536])
# f.close()

dev.__del__
out.release()