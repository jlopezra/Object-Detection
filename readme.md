
**Problem:**
The goal is to train a model with at most 4.5 million trainable parameters which determines whether each image has a star and, if so, finds a rotated bounding box that bounds the star.

More precisely, the labels contain the following five numbers, which the model will predict:
* the x and y coordinates of the center
* yaw
* width and height.

If there is no star, the label consists of 5 `np.nan`s. The height of the star is always noticeably larger than its width, and the yaw points in one of the height directions. The yaw is always in the interval `[0, 2 * pi)`, oriented counter-clockwise and with zero corresponding to the upward direction.

**Evaluation:**
The metric is the percent of correctly identified stars based on an IOU threshold of 0.7 (for 1024 random samples).

`compute_score.py` file that reproduces the reported score
`train.py` script that reproduces a model which produces the final score
`requirements.txt` file that includes all python dependencies and their versions
Look closely at examples of the data using `display_example.py`
Look at how the data is synthesized in `utils.py`



