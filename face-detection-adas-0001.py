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
    'streams': ['metaout'],
    'ai':
    {
        'blob_file': 'face-detection-adas-0001.blob',
        'blob_file_config': 'face-detection-adas-0001.json'
    }
}


# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline(configs)

if p is None:
    print('Pipeline is not created.')
    exit(2)

while True:

    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

    for i, nnet_packet in enumerate(nnet_packets):
        print(nnet_packet)
        print(nnet_packet.entries())
        # the result of the MobileSSD has detection rectangles (here: entries), and we can iterate threw them
        for i, e in enumerate(nnet_packet.entries()):
            # for MobileSSD entries are sorted by confidence
            # {id == -1} or {confidence == 0} is the stopper (special for OpenVINO models and MobileSSD architecture)
            print(e)
            # for label in ["image_id", "label", "conf", "x_min", "y_min", "x_max", "y_max"]:
            #     print(label, e[0][label])

    # for i, nnet_packet in enumerate(nnet_packets):
    #     detections = []
    #     print(nnet_packet)
    #     print(nnet_packet.entries())
    #     # print(len(nnet_packet.entries()[0])) # failing w/o json file .. required?
    #     for i in range(len(nnet_packet.entries()[0][0])):
    #         detections.append(nnet_packet.entries()[0][0][i])
    #         print(nnet_packet.entries()[0][0][i]['label'])
    #     # entries_prev = detections

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

            frame = cv2.resize(frame, (300, 300))
            cv2.imshow('previewout', frame)




    if cv2.waitKey(1) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.

print('py: DONE.')
