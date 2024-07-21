# Balancing the non-cancer images to make them equal to Cancer Images (500 in number)

import os
import imageio
import random
import albumentations as A

# Paths to the folders containing images
non_cancer_folder = r"C:\Users\admin\Documents\MehtA+ Projects\Smile-Savior\New-Data\OC_Dataset_kaggle_new\NON_CANCER"
output_non_cancer_folder = r"C:\Users\admin\Documents\MehtA+ Projects\Smile-Savior\New-Data\OC_Dataset_kaggle_new\Augmented_non_cancer"

# Augmentation sequence
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),  # horizontal flips
    A.RandomRotate90(p=0.5),  # random rotations
    A.RandomBrightnessContrast(p=0.2),  # change brightness and contrast
    A.GaussianBlur(p=0.2)  # blur
])

def augment_images(image_folder, output_folder, num_augmented):
    images = os.listdir(image_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Randomly shuffle images to apply different augmentations
    random.shuffle(images)
    
    # Calculate how many more images are needed
    current_images_count = len(images)
    additional_needed = num_augmented - current_images_count
    
    print(f"Current image count: {current_images_count}")
    print(f"Additional images needed: {additional_needed}")
    
    if additional_needed <= 0:
        print("No augmentation needed, sufficient images available.")
        return
    
    augmented_images_count = 0

    # Augment existing images to reach the desired count
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        img = imageio.imread(img_path)
        
        # Apply augmentation
        for i in range(additional_needed):  # Adjust the number of augmentations
            augmented_img = augmenter(image=img)['image']
            augmented_img_name = f"aug_{augmented_images_count}_{img_name}"
            augmented_img_path = os.path.join(output_folder, augmented_img_name)
            imageio.imwrite(augmented_img_path, augmented_img)
            augmented_images_count += 1
            if augmented_images_count >= additional_needed:
                break
        if augmented_images_count >= additional_needed:
            break

    print(f"Augmented images count: {augmented_images_count}")

# Augment non-cancer images to reach 500
num_augmented = 500 - 450  # Calculate how many more images are needed
augment_images(non_cancer_folder, output_non_cancer_folder, num_augmented)

print("Non-cancer images balancing attempt completed.")


