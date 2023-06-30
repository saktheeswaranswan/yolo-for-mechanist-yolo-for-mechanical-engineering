import cv2
import numpy as np

# Load YOLOv3-tiny model
net = cv2.dnn.readNet("yolov3-tiny-best_14000ppe.weights", "yolov3-tiny-objppe.cfg")

# Load class names
with open("ppe.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set confidence threshold and non-maximum suppression threshold
conf_threshold = 0.5
nms_threshold = 0.4

# Get the output layer names of the network
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to perform object detection
def detect_objects(image, trained_object=None):
    height, width, _ = image.shape

    # Create a blob from the input image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Forward pass through the network
    outs = net.forward(output_layers)

    # Initialize empty lists for bounding boxes, class IDs, and confidences
    boxes = []
    class_ids = []
    confidences = []

    # Iterate over each output layer
    for out in outs:
        # Iterate over each detection
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions based on confidence threshold
            if confidence > conf_threshold:
                # Scale the bounding box coordinates to the original image size
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")

                # Calculate the top-left corner coordinates
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                # Add the bounding box, class ID, and confidence to the respective lists
                boxes.append([x, y, int(w), int(h)])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    # Perform non-maximum suppression to eliminate overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Iterate over the selected bounding boxes and draw them on the image
    for i in indices:
        i = i
        box = boxes[i]
        x, y, w, h = box

        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{classes[class_ids[i]]}: {w}x{h}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Object Detection", image)

# Function to capture video from webcam and perform object detection
def perform_object_detection(trained_object=None):
    cap = cv2.VideoCapture("ppetest.mp4")

    while True:
        ret, frame = cap.read()
        detect_objects(frame, trained_object)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

# Get the trained object from user input
trained_object = input("Enter the trained object: ")

# Perform object detection on the live video stream
perform_object_detection(trained_object)

