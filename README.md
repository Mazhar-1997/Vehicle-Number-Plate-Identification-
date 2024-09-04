# Car Number Plate Recognition
This project is a Python-based implementation of car number plate recognition using OpenCV and Tesseract OCR. The system captures video from a webcam, detects car number plates using Haar Cascades, and performs Optical Character Recognition (OCR) to extract the text from the detected plates. The extracted text can be saved and processed further, such as storing it in a CSV file for record-keeping.

# Features
Live Video Feed: Captures video in real-time from the webcam.
Number Plate Detection: Uses Haar Cascades to detect number plates in the video stream.
Optical Character Recognition (OCR): Utilizes Tesseract OCR to extract text from detected number plates.
Save Detected Plates: Allows the user to save images of detected number plates and the recognized text.
Customizable Video Feed: The size and position of the video feed window can be adjusted, and the window is set to always be on top.
# Installation
Install the required packages:
pip install opencv-python pytesseract
Install Tesseract OCR:
Download and install Tesseract-OCR.
Add the Tesseract executable path to your system's PATH environment variable.
Download the Haar Cascade for Russian Plate Number Detection:

Ensure that the Haar Cascade file (haarcascade_russian_plate_number.xml) is in the correct directory.
Usage
python car_plate_recognition.py
Interact with the live video feed:

# Detect and display number plates in real-time.
Press 's' to save the detected number plate image and print the recognized text in the terminal.
Press 'q' to quit the video feed and close all windows.
# Customization:
Adjust the video feed's width and height by modifying the cap.set() parameters in the script.
Modify the Haar Cascade path and other parameters as needed.
Project Structure
car_plate_recognition.py: Main script for detecting and recognizing number plates.
haarcascade_russian_plate_number.xml: Haar Cascade file for detecting Russian-style number plates.
plates/: For plate images will be stored.


Future Enhancements
CSV Export: Implement a feature to save the recognized text in a CSV file.
Multi-Language OCR: Expand OCR capabilities to support multiple languages.
Enhanced Plate Detection: Improve the accuracy of plate detection under various lighting conditions.


# License
This project is licensed under the MIT License - see the LICENSE file for details.


