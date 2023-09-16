# Presentation
[Google Slides Presentation](https://docs.google.com/presentation/d/1yHWKIPSJZeGnE_Z-sF241x8z34RcwTYgR_yRmU15R1g/edit?usp=sharing) 
# PatternRecognition
Pattern recognition using OpenCV
## Requirement Analysis
There are two different patterns in the following image: a square and a circle. As a human, I can recognize these two patterns without any extra effort. However, a computer is not the same. This project aims to instruct how to ask a computer to recognize patterns in an image using Python programming language and OpenCV library.   

🎯 Ask the computer to recognize the patterns in the following image using Python and OpenCV:  

![e_noise](https://github.com/RuichenCN/PatternRecognitionProject/assets/113652310/69186de7-3973-4baf-8bdf-66f66f1b0bc6)

## Design
1. Noise reduction by Blurring  
   Noise reduction is used to distinguish the edge of each objects in the image. We can see a lot of salt and pepper noises in this image. So, the first step we need to implement is to clean this image.
3. Histogram Analysis  
    A histogram is a graphical representation of the distribution of pixel intensities in an image. We use it to determine the threshold value that will be used for the next step.
4. Otsu’s Thresholding  
    Otsu's thresholding is an image segmentation technique used to automatically determine an optimal threshold value to separate pixels of an input grayscale image into two classes: foreground and background. The goal is to find the threshold value that minimizes the intra-class variance while maximizing the inter-class variance.  
5. Connectivity Analysis  
    Connectivity analysis is a process to identify and analyze connected regions or components within an image. It can distinguish separate objects in an image by using either a 4-pixel or 8-pixel connectivity. 4-pixel connects pixels with the same value along the edges. 8-pixel does the same with the addition of edges and corners.   
6. Pattern Recognition  
    The final step is to recognize and categorize unknown patterns in the image.
## Implementation
#### Preparation
* My System Info:  
  MacBook Air 2020  
  Mac OS 13.4  
  Chip M1   
  Memory 8GB

* Install OpenCV  
  1. Import this code in your terminal:
    ```
    pip install opencv-python
    ```
  2. Verify if installation is successful
     * Open a text editor (such as VSCode, Sublime Text, or Atom) and create a new file.
     * Add the following code to the file:
         ```python
         import cv2

         print("OpenCV version:", cv2.__version__)
         ```
     * Run the Python code
     * If the installation is successful, the script will print the version of OpenCV installed on your system.
         ```python
         OpenCV version: 4.8.0
         ```
#### Step1: Noise reduction by blurring
* Read the JPG image
  Load the JPG image using OpenCV's imread() function
   ```python
   import cv2
  
   # Read the JPG image
   original_image = cv2.imread('path_to_your_image.jpg')
   ```
* Convert the image to grayscale
  The image hasn't been in grayscale. You should convert it using the cvtColor() function first before further processing.
  
  ```python
  # Convert the image to grayscale
  gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
  ```
* Apply Gaussian filter for denoising
  
  Use the GaussianBlur() function to apply the Gaussian filter for denoising. The filter size (kernel size) and standard deviation (sigma) are essential parameters to control the strength of the denoising effect. Experiment with different values to get the desired result.
  ```python
  # Denoising
  denoised_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)
  ```
  My practical arguments:
  
  ```python
  # Arguments
  kernel_size = 5
  sigma = 2
  ```
So far, Step1 completed🎉. We get a denoised image ready for further processing. Let's continue.
#### Step2: Histogram analysis
* Intall matplotlib  
  Import the following code into the terminal
  ```
  pip install matplotlib
  ```
* Use matplotlib to create a code to plot histogram
  ```python
  import cv2
  import numpy as np
  from matplotlib import pyplot as plt
  img = cv2.imread('denoised_image.jpg')
  plt.hist(img.ravel(),256,[0,256])
  plt.show()
  ```
  > Code explanation:  
  > **plt.hist()** finds the histogram and plot it  
  > **ravel()** changes 2D/multi-dimensional array to contiguous flattened array  
  > **256** = number of pixels
  
  What you will get:  
  ![his](https://github.com/RuichenCN/PatternRecognitionProject/assets/113652310/c01fa492-7ca6-44f3-8d6b-7bf18c6359bc)
#### Step3: Otsu’s thresholding
In this step, both Global Thresholding and Otsu's Thresholding are applied to compare their effectiveness on the given image. 

1. Global thresholding to the gray original image     
   Global thresholding is a simple thresholding method that divides the entire image into two regions: pixels with values below or equal to the threshold (set to 0) and pixels with values above the threshold (set to 255).  
    ```python
    # Global thresholding
    th_img1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)[1]
    ```
    > Code explanation:   
    > **threshold()** applies to get a binary image out of a grayscale image.

2. Otsu's thresholding to the gray original image  
    ```python
    # Otsu's thresholding
    th_img2 = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    ```
3. Otsu's thresholding to the Gaussian filtered image  
    ```python
    # Otsu's thresholding after Gaussian filtering
    th_img3 = cv2.threshold(denoised_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    ```
4. Plot all the images and their histograms   
    ```python
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
    ```
    > Code explanation:   
    > **THRESH_BINARY**  
    > <img width="295" alt="binary" src="https://github.com/RuichenCN/PatternRecognitionProject/assets/113652310/724429ad-2557-404c-a967-05a00cda5565">   
    > **THRESH_OTSU** uses the Otsu algorithm to choose the optimal threshold value. It is implemented only for 8-bit single-channel images.  

5. What you will get:  
   ![Figure_1](https://github.com/RuichenCN/PatternRecognitionProject/assets/113652310/1941531c-e887-4299-af27-02fd8277ebcf)
#### Step4: Connectivity analysis
* Use this function and the *th_img3* we get in the last step:

  ```python
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
  ```

  > Code explanation:    
  > The above uses a classical algorithm with makes two passes. First to record equivalences and assign temporary labels. Second to replace the temporary labels with the label of its equivalent class. 


* What you will get:  
![Figure_2](https://github.com/RuichenCN/PatternRecognitionProject/assets/113652310/e0ea674c-5083-47be-bdc9-6079e5f7fca5)  
Different colors represent different patterns
#### Step5: Pattern Recognition
* Intall cvlib  
  Import the following code into the terminal
  ```
  pip install cvlib
  ```
* Import this code into your py file   
  ```python
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
  ```
  > Code explanation:    
  > First, we use the *th_img3* to find all of the contours. The contours tell us the coordinates for each shape.       
  > Then, we can find the name of the shape of the object by approximating the contours.   
  > Last step, we convert the gray image into the color one and draw the contours in the denoised image.   

* What you will get:   
  <img width="560" alt="Screenshot 2023-08-06 at 12 31 59 PM" src="https://github.com/RuichenCN/PatternRecognitionProject/assets/113652310/27a8c001-fd3a-4146-a5bc-fee7a616a7d2">

## References
  [Gaussian Filter](https://en.wikipedia.org/wiki/Gaussian_filter)  
  [Histogram](https://aishack.in/tutorials/histograms-simplest-complex/)  
  [Threshold](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef7)  
  [Connectivity Analysis](https://iq.opengenus.org/connected-component-labeling/)  
  [Pattern Recognition](https://pysource.com/2018/09/25/simple-shape-detection-opencv-with-python-3/)  
## Acknowledge
  Thank **Dr. Henry Chang** for his detailed guidance   
   Thank you Mikaela Montaos
