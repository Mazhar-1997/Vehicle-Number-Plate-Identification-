import cv2
import os
import pytesseract

# Load the Haar Cascade for Russian plate number detection
harcascade = "C:/Users/Mazhar/OneDrive/Desktop/Projects/car_number_plate_recognition-main/haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

# Initialize video capture for the webcam
cap = cv2.VideoCapture(0)

# Set the width and height of the video feed
cap.set(3, 480)  # width
cap.set(4, 360)  # height

# Minimum area for detecting plates
min_area = 500
count = 0

# Ensure the directory exists
if not os.path.exists("C:/Users/Mazhar/OneDrive/Desktop/Projects/car_number_plate_recognition-main/plates"):
    os.makedirs("C:/Users/Mazhar/OneDrive/Desktop/Projects/car_number_plate_recognition-main/plates")

while True:
    success, image = cap.read()

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect plates in the image
    plates = plate_cascade.detectMultiScale(img_gray, 1.2, 10)

    img_roi = None  # Initialize img_roi to avoid errors if no plates are detected
    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            # Draw a rectangle around the detected plate
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Label the detected area as "Number Plate"
            cv2.putText(image, "Number Plate", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract the region of interest (ROI)
            img_roi = image[y: y + h, x: x + w]
           
            cv2.imshow("ROI", img_roi)
            

    # Display the Live result
    cv2.imshow("Live", image)
    # Create the window and set it to be always on top
    cv2.namedWindow("Live", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Live", cv2.WND_PROP_TOPMOST, 1)
    # Move the window to the left side of the display
    cv2.moveWindow("Live", 1800,500)

    # Save image when 's' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and img_roi is not None:
        cv2.imwrite("C:/Users/Mazhar/OneDrive/Desktop/Projects/car_number_plate_recognition-main/plates/scaned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Plate Saved", (150, 250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", cv2.resize(image,(400,200)))
        cv2.waitKey(1000)
        count += 1
        #for destroy results after wait 1sec.
        #cv2.destroyAllWindows()
        

        # Extract the text and print in terminal, plan to save in csv
        image_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(image_roi_gray)
        print(text)

    #break the loop after 'q' is pressed
    elif key == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
