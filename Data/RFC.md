# Data Acquisition and Preprocessing

## Data Acquisition

Due to the constraint of resources and professional experience, we are unable to obtain D-eye videos from dilated eyes. We then seek for alternative resources online and found:


[![Dilated Adult](https://img.youtube.com/vi/LdGkxhm9PFA/1.jpg)](https://www.youtube.com/watch?v=LdGkxhm9PFA&feature=youtu.be&t=21s)

and

[![Dilated Child](https://img.youtube.com/vi/LdGkxhm9PFA/3.jpg)](https://www.youtube.com/watch?v=LdGkxhm9PFA&feature=youtu.be&t=73s)

Data from D-eye has been received. Special thanks to Spencer Lee from D-eye.
[D-EYE Images and Videos](https://www.dropbox.com/sh/9t9flkrgcc44pmq/AACLLPqsp-6eJSL1bLw0ES86a?dl=0)

## Literature Review

The fragments of video could be stitched using Scale-Invariant Feature Transform. 
[Automatic Panoramic Image Stitching using Invariant Features](http://matthewalunbrown.com/papers/ijcv2007.pdf)

The functionality can be implemented using openCV.
[Introduction to SIFT (Scale-Invariant Feature Transform)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html)

One major concern would be that the retinal blood vessels are not static, making the features unidentifiable for stitching.
Further literature review about approximation algorithms is to be conducted.

## SIFT

Video is parsed into frame images at default interval of 24.
Initial experiment on dummy data. Further preprocessing of the retinal images is required.
