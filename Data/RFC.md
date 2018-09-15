# Data Acquisition and Preprocessing

## Data Acquisition

Due to the constraint of resources and professional experience, we are unable to obtain D-eye videos from dilated eyes. We then seek for alternative resources online and found:


[![Dilated Adult](https://img.youtube.com/vi/LdGkxhm9PFA/1.jpg)](https://www.youtube.com/watch?v=LdGkxhm9PFA&feature=youtu.be&t=21s)

and

[![Dilated Child](https://img.youtube.com/vi/LdGkxhm9PFA/3.jpg)](https://www.youtube.com/watch?v=LdGkxhm9PFA&feature=youtu.be&t=73s)

Inquiry of authorized acquisition of such clips has been sent.

## Literature Review

The fragments of video could be stitched using Scale-Invariant Feature Transform. 
[Automatic Panoramic Image Stitching using Invariant Features](http://matthewalunbrown.com/papers/ijcv2007.pdf)

The functionality can be implemented using openCV.
[Introduction to SIFT (Scale-Invariant Feature Transform)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html)

One major concern would be that the retinal blood vessels are not static, making the features unidentifiable for stitching.
Further literature review about approximation algorithms is to be conducted.
