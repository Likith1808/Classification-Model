import cv2
import numpy as np
import tensorflow as tf

# Load the labels from the labels.txt file
with open('/Users/likith/Documents/Model/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/Users/likith/Documents/Model/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Resize the frame to match the input shape of the model
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0)
    
    # Preprocess the image (normalization)
    input_data = (np.float32(input_data) - 127.5) / 127.5
    
    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run the interpreter
    interpreter.invoke()
    
    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.squeeze(output_data)
    
    # Find the class with the highest score
    max_index = np.argmax(output_data)
    class_label = labels[max_index]
    confidence = output_data[max_index]
    
    # Debugging print
    print(f"Detected: {class_label} with confidence {confidence:.2f}")
    
    # Display the classification result
    cv2.putText(frame, f'{class_label}: {confidence:.2f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
