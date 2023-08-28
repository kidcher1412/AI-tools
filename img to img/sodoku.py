import cv2


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def find_digits(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:
            digit_contours.append(contour)
    return digit_contours


def draw_boxes(image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Load the input image
image_path = 'sudoku_image1.png'
image = cv2.imread(image_path)

# Preprocess the image
preprocessed_image = preprocess_image(image.copy())

# Find digit contours
digit_contours = find_digits(preprocessed_image)

# Draw bounding boxes on digits
image_with_boxes = image.copy()
draw_boxes(image_with_boxes, digit_contours)

# Display the image with bounding boxes
cv2.imshow("Image with Boxes", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
