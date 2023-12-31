import os
import shutil
import face_recognition  # pip install face_recognition
import cv2  # pip install opencv-python
import numpy as np


# Dummy functions: replace these with actual implementations
def is_relevant_image(image_fn):
    image = face_recognition.load_image_file(image_fn)
    face_locations = face_recognition.face_locations(image)
    score = len(face_locations)  # You can modify how the score is calculated based on your needs
    return score

def is_relevant_video(video_fn):
    video_capture = cv2.VideoCapture(video_fn)
    score = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        score += len(face_locations)  # Modify how the score is calculated based on your needs

    video_capture.release()
    return score

# Main function to filter media files
def filter_media_files(parent_folder, ref_score):
    relevant_folder = os.path.join(parent_folder, "relevant_data")
    not_relevant_folder = os.path.join(parent_folder, "not_relevant_data")

    for channel_folder in os.listdir(parent_folder):
        channel_path = os.path.join(parent_folder, channel_folder)
        if os.path.isdir(channel_path) and channel_folder not in ["relevant_data", "not_relevant_data"]:
            for date_folder in os.listdir(channel_path):
                date_path = os.path.join(channel_path, date_folder)
                if os.path.isdir(date_path):
                    for media_file in os.listdir(date_path):
                        media_path = os.path.join(date_path, media_file)
                        if os.path.isfile(media_path):
                            if media_file.lower().endswith(('.png', '.jpg', '.jpeg', 'tiff', 'tif')):
                                score = is_relevant_image(media_path)
                            elif media_file.lower().endswith(('.mp4', '.avi', '.mov')):
                                score = is_relevant_video(media_path)
                            else:
                                continue

                            if score > ref_score:
                                destination_folder = os.path.join(relevant_folder, channel_folder, date_folder)
                            else:
                                destination_folder = os.path.join(not_relevant_folder, channel_folder, date_folder)

                            if not os.path.exists(destination_folder):
                                os.makedirs(destination_folder)
                            shutil.move(media_path, os.path.join(destination_folder, media_file))


def face_bbox(image, face_locations):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding box around each detected face
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Image', rgb_image)

    # Press any key to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Main execution


if __name__ == "__main__":
    parent_folder = "/path/to/your/parent/folder"  # Replace with the path to your parent folder
    ref_score = 0.6  # Replace with your reference score
    # filter_media_files(parent_folder, ref_score)
    path_to_toy = '//home/yakovc/Documents/Data/Telegram_Data_Kobic/gazaalannet/2023_10_05/0_20231005221920.jpg'
    # import face_recognition

    # Load an image
    image = face_recognition.load_image_file(path_to_toy)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    face_bbox(image, face_locations)
