import numpy as np
import cv2
import matplotlib.pyplot as plt


img_ = cv2.imread('/Users/william/Downloads/test.jpg', cv2.COLOR_BGR2RGB)
img_copy = np.copy(img_)
plt.imshow(img_, cmap='gray')

# greenscreen removal happening here
lower_green = np.array([0, 150, 0])
upper_green = np.array([87, 220, 73])

# Masking
mask = cv2.inRange(img_copy, lower_green, upper_green)
masked_image = np.copy(img_copy)
masked_image[mask != 0] = [0, 0, 0]
plt.imshow(masked_image, cmap='gray')


#cv2.imshow('sample image', masked_image)
#cv2.waitKey(0)  # waits until a key is pressed
#cv2.destroyAllWindows()  # destroys the window showing image

# Change background image
background_image = cv2.imread('/Users/william/Downloads/bg.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

# Crop background image to correct size
crop_background = background_image[0:2000, 0:2000]
crop_background[mask == 0] = [0, 0, 0]

final_image = crop_background + masked_image
#plt.imshow(final_image, cmap='gray')
cv2.imwrite('test.jpg', masked_image)
