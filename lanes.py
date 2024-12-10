import cv2
import numpy as np

# Function to calculate the coordinates of the lane lines
def make_coordinates(image, line_parameters):
    # Extract slope and intercept from the line parameters
    slope, intercept = line_parameters
    # Define the y-coordinates for the starting and ending points of the line
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * (3 / 5))  # Set y2 to be 3/5 of the image height, higher up on the image
    # Calculate x-coordinates based on the slope and intercept
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    # Return the coordinates of the line
    return np.array([x1, y1, x2, y2])

# Function to average the slopes and intercepts of the detected lines
def average_slope_intercept(image, lines):
    left_fit = []  # List to store left lane line parameters
    right_fit = []  # List to store right lane line parameters

    # Loop through each line in the lines list
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)  # Reshape to extract x1, y1, x2, y2 values
        # Use numpy.polyfit to calculate slope (m) and intercept (b) of the line
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        # If the slope is negative, it belongs to the left lane, else right lane
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Calculate average slope and intercept for the left and right lane lines
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # Create lines from the averaged parameters for both left and right lanes
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    # Return both left and right lane lines as a numpy array
    return np.array([left_line, right_line])

# Function to apply Canny edge detection on an image
def canny(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Function to draw detected lane lines on the image
def display_lines(image, lines):
    # Create an empty image to draw the lines on
    line_image = np.zeros_like(image)
    # If there are detected lines, draw them on the image
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draw each line with blue color and thickness of 10
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# Function to apply a region of interest mask to the image
def region_of_interest(image):
    # Get the height of the image
    height = image.shape[0]
    # Define a polygon (triangle) to focus on the region of interest for lane detection
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]  # Coordinates of the triangle's vertices
    ])
    # Create a mask to isolate the region of interest
    mask = np.zeros_like(image)
    # Fill the polygon area in the mask with white color
    cv2.fillPoly(mask, polygons, 255)
    # Apply the mask to the image to only keep the region of interest
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Main loop for processing video
cap = cv2.VideoCapture("test2.mp4")  # Open video file
while(cap.isOpened()):
    # Read each frame from the video
    _, frame = cap.read()
    # Apply Canny edge detection
    canny_image = canny(frame)
    # Apply region of interest mask
    cropped_image = region_of_interest(canny_image)
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # Average the detected lines to get smoother lane lines
    averaged_lines = average_slope_intercept(frame, lines)
    # Draw the averaged lane lines on the frame
    line_image = display_lines(frame, averaged_lines)
    # Combine the original frame with the detected lines
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # Display the result
    cv2.imshow("result", combo_image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
