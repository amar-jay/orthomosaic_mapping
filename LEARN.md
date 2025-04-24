In feature detection mainly using OpenCV, upon detection, two components are given the keypoints and the description the keypoint are significant features are easily distinguisable. This is used in detection of movement across frames. and this movement of these keypoints is what is what is classified as movement. In the case below we are using ORBSLAM, it is a framework primarirly in image recognition and robotics with support for monocular cameras as well in detecting position by relative motion as explained.

```python
import cv2
import numpy as np
import os

print("Current working directory:", os.getcwd())

cv2.namedWindow("output", cv2.WINDOW_NORMAL)

img_1 = cv2.imread('./simple_image.jpg')
if img_1 is None:
    print("Error: Image not found or path is incorrect.")
    exit()

orb = cv2.ORB_create(nfeatures=1000)
keypoints_orb, descriptors = orb.detectAndCompute(img_1, None)

output_img = cv2.drawKeypoints(img_1, keypoints_orb, None, color=(255, 0, 255))
cv2.imshow("output", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
``` 


Another alternative is SIFT. SIFT is a much more robust variant of doing so. Unlike ORBSLAM which is a whole SLAM module on its own SIFT is focused on feature matching.

```python
import cv2
import numpy as np
import os

print("Current working directory:", os.getcwd())

cv2.namedWindow("output", cv2.WINDOW_NORMAL)

img = cv2.imread('./simple_image.jpg')
if img_1 is None:
    print("Error: Image not found or path is incorrect.")
    exit()


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT_create()

keypoints = sift.detect(img_gray, None)
cv2.imshow("output", cv2.drawKeypoints(img, keypoints, None, (255, 0, 255)))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Another alternative too is SUFT but couldn't run it.

Great! Now let's move on to the stitching images. As we can see in both sift and orbslam, keypoints are generated. We can match features between two coonsecutive images using K-neearest neighbor and classify it based on a lowe's ratio. okay what is lowe's ratio test. 

$$
text{lowe's ratio} = \frac {\text{best closest match}} {\text{second best closest match}} 
$$


so based on this ratio, we classify what is great and not. So, basically what we are doing is filtering the features from the classification. and and based on that find the approximate distance between frames using trig. TODO: Im definately bad at explaining this. do it again

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_1 = cv2.imread('images/DSC02930.JPG')
# plt.imshow(img_1)
# plt.show()
img_2 = cv2.imread('images/DSC02931.JPG')
# plt.imshow(img_2)
# plt.show()

img1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
plt.imshow(img1)
plt.show()
img2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
plt.imshow(img2)
plt.show()

sift = cv2.SIFT_create() 
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2, k=2)

good = []
for m in matches:
    if (m[0].distance < 0.5*m[1].distance):
        good.append(m)
matches = np.asarray(good)


if (len(matches[:,0]) >= 4):
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
    raise AssertionError('Can’t find enough keypoints.')

dst = cv2.warpPerspective(img_1,H,((img_1.shape[1] + img_2.shape[1]), img_2.shape[0])) #wraped image
dst[0:img_2.shape[0], 0:img_2.shape[1]] = img_2 #stitched image
cv2.imwrite('output.jpg',dst)
plt.imshow(dst)
plt.show()
```
