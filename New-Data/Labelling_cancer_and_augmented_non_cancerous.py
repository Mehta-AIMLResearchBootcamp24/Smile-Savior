import os
import pandas as pd

# Paths to the folders containing images
cancer_folder = r"C:\Users\admin\Documents\MehtA+ Projects\Smile-Savior\New-Data\OC_Dataset_kaggle_new\CANCER"
non_cancer_folder = r"C:\Users\admin\Documents\MehtA+ Projects\Smile-Savior\New-Data\OC_Dataset_kaggle_new\NON_CANCER"
augmented_non_cancer_folder = r"C:\Users\admin\Documents\MehtA+ Projects\Smile-Savior\New-Data\OC_Dataset_kaggle_new\Augmented_non_cancer"

# Function to get image paths and labels
def get_image_paths_and_labels(folder, label):
    images = os.listdir(folder)
    image_paths = [os.path.join(folder, img) for img in images]
    labels = [label] * len(images)
    return image_paths, labels

# Get image paths and labels for cancer and non-cancer images
cancer_images, cancer_labels = get_image_paths_and_labels(cancer_folder, 1)
non_cancer_images, non_cancer_labels = get_image_paths_and_labels(non_cancer_folder, 0)
augmented_non_cancer_images, augmented_non_cancer_labels = get_image_paths_and_labels(augmented_non_cancer_folder, 0)

# Combine the lists
all_images = cancer_images + non_cancer_images + augmented_non_cancer_images
all_labels = cancer_labels + non_cancer_labels + augmented_non_cancer_labels

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame({'image_path': all_images, 'label': all_labels})
df.to_csv('labeled_images.csv', index=False)

print("Data labeling completed successfully. CSV file saved as 'labeled_images.csv'.")
