from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


#סעיף א - הצגה וניתוח של התמונה המקורית 
def analyze_image(input_image_path, output_folder_path):
    input_image = cv2.imread(input_image_path) #טעינת התמונה
    ###grayimg.png###
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_folder_path, 'gray_image.png') 
    cv2.imwrite(gray_path, gray_image) #שמירה
    
    
    ###gray_histogram.png###
    plt.hist(gray_image.ravel(), bins=256, range=[0, 255])
    plt.title('Histogram of Grayscale Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    histogram_path = os.path.join(output_folder_path, 'gray_histogram.png')  
    plt.savefig(histogram_path)# שמירת ההיסטוגרמה


    ###red_histogram.png###
    red_channel = input_image[:, :, 2]  # אדום
    plt.figure()
    plt.hist(red_channel.ravel(), bins=256, range=[0, 255])
    plt.title('Histogram of Red Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    R_histogram_path = os.path.join(output_folder_path, 'red_histogram.png')
    plt.savefig(R_histogram_path)


    ###blue_histogram.png###
    blue_channel = input_image[:, :, 0]  # כחול
    plt.figure()
    plt.hist(blue_channel.ravel(), bins=256, range=[0, 255])
    plt.title('Histogram of blue Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    B_histogram_path = os.path.join(output_folder_path, 'blue_histogram.png')
    plt.savefig(B_histogram_path)


    ###green_histogram.png###
    green_channel = input_image[:, :, 1]  # ירוק
    plt.figure()
    plt.hist(green_channel.ravel(), bins=256, range=[0, 255])
    plt.title('Histogram of green Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    G_histogram_path = os.path.join(output_folder_path, 'green_histogram.png')
    plt.savefig(G_histogram_path)
     
    
    ###gradient_x.png###
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5) #5x5 מסנן בגודל  
    sobelx = cv2.convertScaleAbs(sobel_x)  # המרת התוצאה לערכים  בין 0 עד 255
    gradient_map_path = os.path.join(output_folder_path, 'gradient_map_x.png')
    cv2.imwrite(gradient_map_path, sobelx) #  X  שמירת מפת הגרדיאנטים בכיוון
    
    
    ###gradient_y.png###
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    sobely = cv2.convertScaleAbs(sobel_y)  # המרת התוצאה לערכים  בין 0 עד 255
    gradient_map_y_path = os.path.join(output_folder_path, 'gradient_map_y.png')
    cv2.imwrite(gradient_map_y_path, sobely)    # Y שמירת מפת הגרדיאנטים בכיוון    


    ###gradient_intensity.png###
    intensity = np.sqrt(sobel_x**2 + sobel_y**2)# שורש סכום הנגזרות בריבוע
    GRADintensity = cv2.convertScaleAbs(intensity)    # המרת התוצאה לערכים בין  0 עד 255
    gradient_intensity_path = os.path.join(output_folder_path, 'gradient_intensity.png')
    cv2.imwrite(gradient_intensity_path, GRADintensity)    # שמירת מפת עוצמת הגרדיאנטים

    return gray_image


input_image_path = r'C:\Users\sami3\Downloads\DIP_122.png' 
output_folder_path = r'C:\Users\sami3\Downloads\output_images'
gray_image=analyze_image(input_image_path,output_folder_path)




#סעיף ב - תיקון ושיפור התמונה המקורית 
def first_op(input_image_path,output_folder_path):
    image = Image.fromarray(input_image_path)
    box = (0,0,3600,2800)  #  חיתוך מרובע מהנקודה (0, 0) עד (2800, 3600)
    cropped_image = image.crop(box)    # חתוך את התמונה
    cropped_image_np = np.array(cropped_image)# NumPy המרת התמונה למערך   
    img_firstop = cv2.GaussianBlur(cropped_image_np, (17,17), 10)#טשטוש בעזרת פילטר גאוסי
    img_firstop_path = os.path.join(output_folder_path, 'img_firstop.png')    # שמירת התמונה המעובדת
    cv2.imwrite(img_firstop_path, img_firstop)
    return img_firstop_path    



def second_op(input_image_path, output_folder_path):
    input_image = cv2.imread(input_image_path)    # טען את התמונה
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    img_secondop = cv2.filter2D(input_image, -1, kernel)    #   החלת פילטר החידוד
    img_secondop_path = os.path.join(output_folder_path, 'img_secondop.png')
    cv2.imwrite(img_secondop_path, img_secondop)    # שמירת התמונה אחרי החידוד
    return img_secondop_path
    

img_firstop=first_op(gray_image,output_folder_path)
img_secondop=second_op(img_firstop,output_folder_path)


#סעיף ג - הצגה וניתוח של התמונה המתוקנת 
output_folder_path2 = r'C:\Users\sami3\Downloads\output_images2'
plt.figure()
analyze_image(img_secondop,output_folder_path2)

#סעיף ד - הגרסה הצבעונית 
input_image = cv2.imread(input_image_path) #טעינת התמונה
img_firstop=first_op(input_image,output_folder_path2)
img_secondop=second_op(img_firstop,output_folder_path2)

#סעיף ה -תמונה אחרת 
input_image_path=r'C:\Users\sami3\Downloads\newimg.jpeg'
output_folder_path3=r'C:\Users\sami3\Downloads\output_images3'
input_image = cv2.imread(input_image_path) #טעינת התמונה
img_firstop=first_op(input_image,output_folder_path3)
img_secondop=second_op(img_firstop,output_folder_path3)