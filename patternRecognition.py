import cv2
import numpy as np
from matplotlib import pyplot as plt


# Read the JPG image
original_image = cv2.imread('original_image.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Denoising
kernel_size = 5
sigma = 2
denoised_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)

# Plot histogram using matplotlib
plt.hist(denoised_image.ravel(),256,[0,256])   # finds the histogram and plot it
plt.show()

# Global thresholding
th_img1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)[1]

# Otsu's thresholding
th_img2 = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# Otsu's thresholding after Gaussian filtering
th_img3 = cv2.threshold(denoised_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# plot all the images and their histograms
images = [gray_image, 0, th_img1, gray_image, 0, th_img2, denoised_image, 0, th_img3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding(v=127)', 'Original Noisy Image','Histogram',"Otsu's Thresholding", 'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

# Connectivity analysis
def connected_component_label(img):
    # Applying cv2.connectedComponents()
    num_labels, labels = cv2.connectedComponents(img)
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue==0] = 0
    #Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()
connected_component_label(th_img3)

# Pattern Recognition
font = cv2.FONT_HERSHEY_COMPLEX
contours = cv2.findContours(th_img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(denoised_image, [approx], 0, (0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 4:
        cv2.putText(denoised_image, "Square", (x,y), font, 1, (255))
    else:
        cv2.putText(denoised_image, "Circle", (x,y), font, 1, (255))
img = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img, contours, -1, (255,0,0), 2)
cv2.imshow("Pattern recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()