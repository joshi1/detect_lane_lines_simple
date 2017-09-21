#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
#%matplotlib inline
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hsvscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

def histequalize(img):
    return cv2.equalizeHist(img)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def gather_lines_info(lines, vertices):
    #my_print (vertices)
    #my_print ("x1={}, y1={}".format(vertices[0][1][0], vertices[0][1][1]))
    #my_print ("x2={}, y2={}".format(vertices[0][2][0], vertices[0][2][1]))
    lines_info = []
    slopes_info = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if(x1 != x2):
                slope = (y2-y1)/(x2-x1)
                b = y1 - slope*x1
                if(slope):
                    new_y1 = int(vertices[0][1][1])
                    new_x1 = int((new_y1-b)/slope)
                    new_y2 = int(vertices[0][3][1]) # max_y i.e. height
                    new_x2 = int((new_y2-b)/slope)
                    lines_info.append([slope, b, new_x1, new_y1, new_x2, new_y2])   
                    slopes_info.append(slope)

    columns = ['slope', 'intercept', 'x1', 'y1', 'x2', 'y2']
    df = pd.DataFrame(lines_info, columns=columns)
    return df

def sanity_check_line_std_based_outliers(data):
    global std_m 
    return abs(data - np.mean(data)) >= m * np.std(data)

def sanity_check_line_mad_based_outliers(data):
    global mad_thresh
    if len(data.shape) == 1:
        data = data[:,None]
    median = np.median(data, axis=0)
    my_print("mad: median {}".format(median))

    diff = np.sum((data - median)**2, axis=-1)
    diff = np.sqrt(diff)
    my_print("mad: diff {}".format(diff))
    
    med_abs_deviation = np.median(diff)
    my_print("mad: dev {}".format(med_abs_deviation))
    
    modified_z_score = 0.6745 * diff / med_abs_deviation
    my_print("mad: z_score {}".format(modified_z_score))

    my_print("mad: Final {}".format(modified_z_score > mad_thresh))

    return modified_z_score > mad_thresh

def sanity_check_line_slope(line):
    #my_print("IN sanity_check_line_slope")
    #my_print(line)
    global min_lslope
    global max_lslope
    global max_rslope
    global min_rslope

    ret = True
    if line.slope >= 0:
        if not(min_lslope < line.slope < max_lslope):
            my_print("Slope out of range {}".format(line.slope))
            ret = False;
    elif line.slope < 0:
        if not(min_rslope < line.slope < max_rslope):
            my_print("Slope out of range {}".format(line.slope))
            ret = False
    return ret

def sanity_check_group(group):
    # Sanity check the group
    #my_print ("*** IN sanity_check_group")
    #my_print (group)
    
    lines_info = []
    for ix, line in group.iterrows():
        #my_print(line)
        ret = sanity_check_line_slope(line)
        if (ret == True):
            lines_info.append(line)
        else:
            my_print("Ignoring line {}".format(line.slope))
            
    new_group = pd.DataFrame(lines_info, columns=group.columns)

    new_group.reset_index(drop=True, inplace = True)

    slopes = new_group['slope']

    #Check1: Detect and delete outliers 
    #outlier = sanity_check_line_std_based_outliers(slopes)
    outlier = sanity_check_line_mad_based_outliers(slopes)
    
    #my_print (outlier)
    for i in range(0, len(outlier)):
        if( outlier[i] == True):
            my_print ("**** Found outlier {}".format(i))
            new_group.drop(new_group.index[[i]])

    #my_print (group)
    #my_print ("***Out sanity_check_group")

    new_group.reset_index(drop=True, inplace = True)
    return new_group

def process_lines(img, vertices, a, rl):
    new_grp = sanity_check_group(a)
        
    m = new_grp.mean()
    if(math.isnan(m.slope) == False):
        
        my_print ("mean {}", m)
        b = m.y1 - m.slope*m.x1
        y1 = int(vertices[0][1][1])
        x1 = int((y1 - b)/m.slope)
        y2 = int(vertices[0][3][1]) # max_y i.e. height
        x2 = int((y2 - b)/m.slope)
        if PROCESSING_VIDEO == True:
            global first_frame
            global prev_right
            global prev_left
            global weight
            
            if(first_frame == False):
                #my_print("modify image")
                if(rl == "right"):
                    my_print("Right: {}".format(prev_right))
                    new = np.array([x1, y1, x2, y2])*weight + \
                          np.array(prev_right)*(1-weight)
                else:
                    my_print("Left: {}".format(prev_left))
                    new = np.array([x1, y1, x2, y2])*weight + \
                          np.array(prev_left)*(1-weight)
                    
                x1 = new[0]
                y1 = new[1]
                x2 = new[2]
                y2 = new[3]
                
            if(rl == "right"):
                prev_right = [x1, y1, x2, y2]
            else:
                prev_left = [x1, y1, x2, y2]
                
        cv2.line(img,\
                 (int(x1), int(y1)), \
                 (int(x2), int(y2)), \
                 [255,0,0], 9)


def create_line_groups(img, df, vertices):
    global prev_left
    global prev_right
    global first_frame
    #group into negative slope and positive slope
    df.sort_values(by= 'slope', inplace = True)

    l_lines = df.loc[df['slope'] >= 0]
    r_lines = df.loc[df['slope'] <0]

    my_print("Processing Left line")
    my_print(l_lines)
    process_lines(img, vertices, l_lines, "left")
    
    my_print("Processing Right line")
    my_print(r_lines)
    process_lines(img, vertices, r_lines, "right")
    if(first_frame == True):
        first_frame = False

            
def hough_lines(img, vertices, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    global prev_img
    if HOUGHLINES_P:
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    else:
        lines = cv2.HoughLines(img, rho, theta, 200)
    

    df = gather_lines_info(lines, vertices)

    #my_print("hough_lines: Shape {}".format(img.shape))
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)

    create_line_groups(line_img, df, vertices)
    
    #line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    #draw_lines(line_img, lines, vertices)
    
    line_img_gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

    return line_img, line_img_gray

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def show_img(img, title = "No Title", cmap=plt.rcParams['image.cmap']):
    if DEBUG_ON:
        #plt.interactive(False)
        plt.title(title)
        plt.imshow(img, cmap=cmap)
        plt.show()

def my_print(*arg):
    if DEBUG_ON:
        print(arg)
        
def set_global_variables():
    #Gaussian Blur
    global kernel_size
    kernel_size = 5 #original = 5
    
    #Canny
    global low_threshold
    global high_threshold
    low_threshold = 1
    high_threshold = 150 #original = 150

    #region of interest
    global toplencol_offset
    global toplenrow_offset
    toplencol_offset = 50
    toplenrow_offset = 60

    #Hough transform
    global rho # distance resolution in pixels of the Hough grid
    global theta # angular resolution in radians of the Hough grid
    global threshold  # minimum number of votes (intersections in Hough grid cell)
    global min_line_length #minimum number of pixels making up a line
    global max_line_gap  # maximum gap in pixels between connectable line segments
    rho = 1
    theta = np.pi/180 
    threshold = 10
    min_line_length = 10
    max_line_gap = 50

    #Final line weighting
    global α #not using now
    global β #not using now
    global λ #not using now

    α=0.8 #not using now
    β=1. #not using now
    λ=0. #not using now

    #Sanity check lines
    global min_lslope
    global max_lslope
    global max_rslope
    global min_rslope
    min_lslope = 0.4
    max_lslope = 0.8
    max_rslope = -0.4
    min_rslope = -0.8

    #Sanity check std
    global std_m
    std_m = 1 #original = 1.5
    global mad_thresh
    mad_thresh = 0.8
    
    #Stabilizing
    global first_frame
    first_frame = True
    global prev_right
    global prev_left
    global weight
    prev_left = []
    prev_right = []
    weight = 0.2
    

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    # Read in and grayscale the image
    global kernel_size
    global low_threshold
    global high_threshold
    global toplencol_offset
    global toplenrow_offset
    global rho
    global theta
    global threshold
    global min_line_length
    global max_line_gap
    global α
    global β
    global λ
    
    #plt.interactive(False)

    #*** Tried for challenge video. didnt quite work too well by itself.
    #*** Would have had to use colors etc. Didnt think that made sense!
    #img_hsv = hsvscale(image)
    #show_img(img_hsv, title="HSV scale image")
    
    #Convert image to grayscale
    img_gray = grayscale(image)
    show_img(img_gray, cmap='gray', title="Gray scale image")

    # *** Tried for challenge. For brightness of the image. Worked somewhat!
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_cl1 = clahe.apply(img_gray)
    show_img(img_cl1, cmap ='gray', title="CLAHE image")

    # *** Tried for challenge video. Didnt work at all!    
    #img_eq = histequalize(img_gray)
    #show_img(img_eq, cmap ='gray', title="Equalized image")

    # *** Tried for challenge video. Didnt work too well!    
    #dst_img = img_gray.copy()
    #Normalize image
    #img_n = cv2.normalize(img_gray, dst_img, alpha=0, beta=1, \
    #                      norm_type=cv2.NORM_MINMAX,\
    #                      dtype=cv2.CV_32F)
    #show_img(img_n, cmap='gray', title="Normalized Image")
    
    # Define a kernel size and apply Gaussian smoothing
    img_blur_gray = gaussian_blur(img_gray, kernel_size)
    show_img(img_blur_gray, cmap='gray', title="Blur image")
    
    # Define parameters for Canny and apply
    img_edges = canny(img_blur_gray, low_threshold, high_threshold)
    show_img(img_edges, cmap='gray', title="Cannied image")

    # Define a four sided polygon to mask
    imshape = image.shape

    my_print("image.shape: {}".format(image.shape))
    
    if False:
        vertices = np.array([[(0,imshape[0]),\
                              (imshape[1]/2-toplencol_offset, imshape[0]/2 + toplenrow_offset), \
                              (imshape[1]/2+toplencol_offset, imshape[0]/2 + toplenrow_offset), \
                              (imshape[1],imshape[0])]], \
                            dtype=np.int32)
    else:
        #** A new polygon to eliminate the region between lanes as much as possible
        vertices = np.array([[(0,imshape[0]),\
                              (imshape[1]/2-toplencol_offset, imshape[0]/2 + toplenrow_offset), \
                              (imshape[1]/2+toplencol_offset, imshape[0]/2 + toplenrow_offset), \
                              (imshape[1],imshape[0]),
                              (imshape[1]-150, imshape[0]),
                              (imshape[1]/2+toplencol_offset-150, imshape[0]/2 + toplenrow_offset+150), \
                              (imshape[1]/2-toplencol_offset-150, imshape[0]/2 + toplenrow_offset+150), \
                              (0+150,imshape[0])]], \
                            dtype=np.int32)

    img_interest = region_of_interest(img_edges, vertices)
    show_img(img_interest, cmap='gray',title="Area of interest")
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on

    # Output "lines" is an array containing endpoints of detected line segments
    [mat_img, img_lines] = hough_lines(img_interest, vertices, rho, theta, \
                           threshold, min_line_length, max_line_gap)

    show_img(mat_img, cmap='gray',title="Hough transformed image")
    
    #img_interest = region_of_interest(mat_img, vertices)
    #show_img(img_interest, cmap='gray',title="Area of interest2")
    
    # Experiment: Try to apply hugh transform again
    if False:
        threshold2 = 10
        min_line_length2 = 10
        max_line_gap2 = 150
    
        [mat_img, img_lines2] = hough_lines(img_lines, rho, theta, \
                               threshold2, min_line_length2, max_line_gap2)
    
        show_img(img_lines2, cmap='gray',title="Hough transformed image 2")
    
    img_final = weighted_img(mat_img, image)

    return img_final

DEBUG_ON = False
HOUGHLINES_P = True
PROCESSING_VIDEO = True
IMG_DIFF = False

set_global_variables()

if PROCESSING_VIDEO == False:
    import os
    test_images = os.listdir("test_images/")

    my_print ("Test images in test_images directory {}".format(test_images))
    for test_image in test_images:
        img = mpimg.imread("test_images/"+test_image);
        my_print ("Begin Processing img {}".format(test_image))
        show_img(img, title="Initial Image")
        final = process_image(img);
        show_img(final, title="Final Image")
        break
    
else:
    if True:
        white_output = 'output_white.mp4'
        clip1 = VideoFileClip("solidWhiteRight.mp4")
        
        #my_print ("Clip1 Size {}".format(clip1.size))
        
        #fps = clip1.fps
        #foi_sec = 150/fps
        #cut_clip1 = clip1.subclip(foi_sec, foi_sec+3/fps)
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile(white_output, audio=False)
        
        yellow_output = 'output_yellow.mp4'
        clip2 = VideoFileClip('solidYellowLeft.mp4')
        yellow_clip = clip2.fl_image(process_image)
        yellow_clip.write_videofile(yellow_output, audio=False)
    
        challenge_output = 'output_.mp4'
        clip2 = VideoFileClip('challenge.mp4')
        challenge_clip = clip2.fl_image(process_image)
        challenge_clip.write_videofile(challenge_output, audio=False)
