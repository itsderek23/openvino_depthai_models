#!/usr/bin/env python3

import sys
from time import time
from time import sleep
import numpy as np
import cv2

import depthai

import consts.resource_paths


cmd_file = consts.resource_paths.device_cmd_fpath
if len(sys.argv) > 1 and sys.argv[1] == "debug":
    cmd_file = ''
    print('depthai will not load cmd file into device.')

if not depthai.init_device(cmd_file):
    print("Error initializing device. Try to reset it.")
    exit(1)


configs = {
    'streams': ['metaout','previewout'],
    'ai':
    {
        'blob_file': 'face-detection-retail-0004.blob',
        'blob_file_config': 'face-detection-adas-0001.json' # uses adas config as I believe they are the same.
    }
}


# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline(configs)

if p is None:
    print('Pipeline is not created.')
    exit(2)

entries_prev = []

while True:

    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

    for i, nnet_packet in enumerate(nnet_packets):
        # https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_retail_0004_description_face_detection_retail_0004.html#outputs
        # Shape: [1, 1, N, 7], where N is the number of detected bounding boxes
        for i, e in enumerate(nnet_packet.entries()):
            for label in ["image_id", "label", "conf", "x_min", "y_min", "x_max", "y_max"]:
                print(label, e[0][label])
            print("---")
            if e[0]['conf'] == 0.0:
                break

            if i == 0:
                print("clear!")
                entries_prev.clear()

            # save entry for further usage (as image package may arrive not the same time as nnet package)
            entries_prev.append(e[0])

    for packet in data_packets:
        if packet.stream_name == 'previewout':
            data = packet.getData()
            # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = data[0,:,:]
            data1 = data[1,:,:]
            data2 = data[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            # iterate threw pre-saved entries & draw rectangle on image:
            for e in entries_prev:
                # the lower confidence threshold - the more we get false positives
                if e['conf'] > 0.5:
                    x1 = int(e['x_min'] * img_w)
                    y1 = int(e['y_max'] * img_h)

                    pt1 = x1, y1
                    pt2 = int(e['x_max'] * img_w), int(e['y_min'] * img_h)
                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255))

            frame = cv2.resize(frame, (300, 300))
            cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.

print('py: DONE.')
