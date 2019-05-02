---
title:  "Intersection Over Union aka IoU"
tags: [iou]
---

# Intersection Over Union aka IoU

A lot of blog post already available which are related to the IoU topic. Current post does not pretend to be treasure chest of  but here I've wrote down couple of notes which helped me understand the algorithm better and implement it by myself.

## Goal
1. Understand idea behind the algorithm
2. Implement algorithm from scratch

## Idea behind the algorithm
The aim of IoU algorithm is to show how bound box(`A`) is similar or different from bound box (`B`). IoU produce one value in range from 0 to 1. Where 0 means totally different bound boxes and 1 means - this is the same bound box. In other words IoU algorithm is just a similarity function with 2 arguments `A` and `B` that returns float value.

We could define this function in next way:

```python
def iou(bbox_a:list, bbox_b:list)->float:
    # calculate IoU here
    pass
```

Let's define two variable corresponding to bounding boxes and use `[xmin, xmax, ymin, ymax]` convention for bound box definition:

```python
bbox_a = [0, 512, 0, 512]
bbox_b = [0, 256, 0, 256]
```

Each bbox could be shown as rectangle area which could be calculated as `rectangle_width * rectangle_height`. Height and width could be calculated from bbox(e.g. `xmax-xmin`) if necessary, but let's skip it for a current moment.

Intersection is the area where two bboxes are overlapped.

Based on algorithm name we have to find 2 values:
- ***intersection*** of two bboxes(the area where two bboxes are overlapped)
- ***union*** of two bboxes(bbox areas sum - intersection)

We know how to calculate area of the rectangle, but it will give us only sum of bbox areas but not intersection nor union.

## Implement algorithm from scratch
If we find ***intersection*** we will have all necessary for algorithm implementation. We could make it in different ways, but here is one of most elegant ways in my opinion[[1](https://github.com/experiencor/keras-yolo2/blob/4e8c85ce02435f136d4f4cfe930b4ccb759fbaf8/utils.py#L182)]

<!-- add image here -->
Based on image above we could have two ways how bboxes are related to each other:

1. `bbox_b` is lefter than `bbox_a`
2. `bbox_b` is righter than `bbox_a`

This way is right but it does not save us when boxes are righter or lefter but not overlapped. Let's fix it:

1. `bbox_b` is lefter than `bbox_a`
    1. bboxes area are not overlapped
    2. bboxes area are overlapped
2. `bbox_b` is righter than `bbox_a`
    1. bboxes area are not overlapped
    2. bboxes area are overlapped

But wait, boxes also could be higher and lower to each other. That is true. Let decrease our space from 2D to 1D and make calculations only for one axis and rewrite statements above which will better reflect process. If we work only with one axis we can't treat values as lefter or righter any more, because it could leads to misunderstanding in 2D space in feature. Let use ***smaller*** and ***bigger*** words instead. Also let change ***bbox*** to ***value*** and ***box area*** to ***interval***.

1. `value_b` is smaller than `value_a`
    1. intervals are not overlapped
    2. intervals are overlapped
2. `value_b` is bigger than `value_a`
    1. intervals are not overlapped
    2. intervals are overlapped

Go ahead and create new our function in code which will reflect statements above:

```python
def get_axis_overlap(v1:int, v2:int, v3:int, v4:int)->int:
    # v values corresponds to both bbox values on the same axis.
    # For example at X axis it will be x1, x2, x3, x4 respectively
    # x1 and x2 corresponds to bbox_a and x3 and x4 corresponds to bbox_b

    # value from bbox_b is smaller than value from bbox_a
    if v3 < v1:
        if v4 < v1:
            # intervals are not overlapped
            pass
        else:
            # bboxes are overlapped
            pass
    # value from bbox_b is bigger than value from bbox_a
    else:
        if v2 < v3:
            # intervals are not overlapped
            pass
        else:
            # bboxes are overlapped
            pass
```

Because we are working on one axis `get_axis_overlap` returns values which corresponds to one of rectangle's sides(e.g. width or height).

In case when there is no overlap we could simply return `0` which basically means that on this axis rectangle side does not exists. In case when there is an overlap we should calculate two values(when smaller and when bigger)

```python
# take biggest axis values for each bboxes v2 for A and v4 for B
# find the smallest between them
intersection_right_border = min(v2, v4)

# at axis intersection_right_border will be bigger(right) value in intersected interval
# take the closes to the intersection_right_border value
# this value will depends on position(smaller, bigger)

# if value from bbox_b is smaller than value from bbox_a
intersection_left_border = v1

# if value from bbox_b is bigger than value from bbox_a
intersection_left_border = v3

intersection_interval_value = intersection_right_border - intersection_right_border

``` 

Let update `get_axis_overlap` function and exclude comments:

```python
def get_axis_overlap(v1:int, v2:int, v3:int, v4:int)->int:
    intersection_right_border = min(v2, v4)
    if v3 < v1:
        if v4 < v1:
            return 0
        else:
            return intersection_right_border - v1
    else:
        if v2 < v3:
            return 0
        else:
            return intersection_right_border - v3
```

>Note: Output is always integer if input is integer which is very helpful when working with pixels

Now we could build all together and make finished iou function:

```python
def iou(bbox_a:list, bbox_b:list)->float:
    x_interval = get_axis_overlap(bbox_a[0], bbox_a[1], bbox_b[0], bbox_b[1])
    y_interval = get_axis_overlap(bbox_a[2], bbox_a[3], bbox_b[2], bbox_b[3])

    intersection = x_interval * y_interval
    w1 = bbox_a[1] - bbox_a[0]
    h1 = bbox_a[3] - bbox_a[2]
    w2 = bbox_b[1] - bbox_b[0]
    h2 = bbox_b[3] - bbox_b[2]

    union = (w1*h1 + w2*h2) - intersection

    iou_ = intersection / union

    return iou_

```

That's it. Let's try to calculate IoU:

```
# Example 1
bbox_a = [0, 512, 0, 512]
bbox_b = [0, 256, 0, 256]

iou(bbox_a, bbox_b)
->0.25

# Example 2
iou(bbox_a, bbox_a)
->1.0

# Example 3
bbox_b = [512, 1024, 512, 1024]
iou(bbox_a, bbox_b)
->0.0

# Example 4
bbox_b = [256, 768, 256, 768]
iou(bbox_a, bbox_b)
->0.14285714285714285

# Example 5
bbox_b = [-256, 256, -256, 256]
iou(bbox_a, bbox_b)
->0.14285714285714285

```

It works as expected. `Example 1` returns 1/4 of the bbox_a as well as `Example 2` because of we use it with the same bbox. It works even with negative values as you can see `Example 4` and `Example 5` have the same intersection area as well as IoU value

I hope this notes could help someone. I have to say that this is my first try in direction of writing posts.

Thank you for reading and see you next blog post ;)
