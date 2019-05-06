---
title: "YOLO v2 steps"
published: false
---

# Steps
1. Preprocessing part
    1. parse dataset annotations
    2. make bbox classifier
        1. Calculate box IoU
        2. define K-means for bboxes
    3. read pretrained YOLO v2 weights
        1. Weight reader
        2. Weight writer
2. Preparing data for training
    1. BBox transformations
    2. get bbox class based on width and height
    3. generating YOLO encoded labels(YOLO encoder)
    4. Put all together at batch generator

3. Additional things
    1. calculating IoU

    Hi Mohammad, as we talked before we will not make speed optimizations in this bid only error fixing. We fixed everything except new News post(I would like to show you or Lisa how to make it, you don't need to hire programer or separate person) and questions I've sent before.