import os
import shutil

# Dummy functions: replace these with actual implementations
def is_relevant_image(image_fn):
    # Replace with actual implementation
    return 0.5

def is_relevant_video(video_fn):
    # Replace with actual implementation
    return 0.5

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

# Main execution
if __name__ == "__main__":
    parent_folder = "/path/to/your/parent/folder"  # Replace with the path to your parent folder
    ref_score = 0.6  # Replace with your reference score
    filter_media_files(parent_folder, ref_score)
