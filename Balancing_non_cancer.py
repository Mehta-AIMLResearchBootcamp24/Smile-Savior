# Balancing the non-cancer images to make them equal to Cancer Images (500 in number)

import imgaug.augmenters as iaa
import os
import imageio.v2 as imageio
import random

# Define folder paths
non_cancer_folder = r"C:\Users\admin\Documents\MehtA+ Projects\Smile-Savior\New-Data\OC_Dataset_kaggle_new\CANCER"
output_non_cancer_folder = r"C:\Users\admin\Documents\MehtA+ Projects\Smile-Savior\New-Data\OC_Dataset_kaggle_new\NON_CANCER"

# Augmentation sequence
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Affine(rotate=(-20, 20)),  # rotations
    iaa.Multiply((0.8, 1.2)),  # change brightness
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # add noise
])

def augment_images(image_folder, output_folder, num_augmented):
    images = os.listdir(image_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Calculate how many more images are needed
    current_images_count = len(images)
    additional_needed = num_augmented - current_images_count
    
    # Randomly shuffle images to apply different augmentations
    random.shuffle(images)
    
    # Augment existing images to reach the desired count
    for i in range(additional_needed):
        img_name = random.choice(images)
        img_path = os.path.join(image_folder, img_name)
        img = imageio.imread(img_path)
        
        # Convert RGBA to RGB if needed
        if img.shape[2] == 4:
            img = img[:, :, :3]
        
        augmented_img = augmenter(image=img)
        augmented_img_name = f"aug_{i}_{img_name}"
        augmented_img_path = os.path.join(output_folder, augmented_img_name)
        imageio.imwrite(augmented_img_path, augmented_img)

# Augment non-cancer images to reach 500
num_augmented = 500  # Desired total number of non-cancer images
augment_images(non_cancer_folder, output_non_cancer_folder, num_augmented)

print("Non-cancer images balanced to 500.")
