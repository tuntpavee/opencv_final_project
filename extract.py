import cv2
import os
import numpy as np
import csv
import random

def add_blur(image):
    k_size = random.choice([3, 5, 7, 9, 11, 15]) 
    return cv2.GaussianBlur(image, (k_size, k_size), 0)

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = random.uniform(10, 60) 
    gauss_noise = np.random.normal(mean, sigma, (row, col, ch))
    return np.clip(image + gauss_noise, 0, 255).astype(np.uint8)

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    blurriness = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_val, std_val = cv2.meanStdDev(gray)
    snr = (mean_val[0][0] / std_val[0][0]) if std_val[0][0] > 0 else 0.0
    return brightness, blurriness, snr

def get_baselines(perfect_folder):
    print("📊 Calculating baselines from perfect_output...")
    blur_values, snr_values = [], []
    
    for filename in os.listdir(perfect_folder):
        if filename.lower().endswith(('.jpg')):
            img = cv2.imread(os.path.join(perfect_folder, filename))
            if img is not None:
                _, blur, snr = extract_features(img)
                blur_values.append(blur)
                snr_values.append(snr)
    
    perf_blur = np.percentile(blur_values, 50) 
    low_blur = np.percentile(blur_values, 5)   
    
    perf_snr = np.percentile(snr_values, 50)
    low_snr = np.percentile(snr_values, 5)
    
    print(f"   -> Blur Thresholds: Perfect >= {perf_blur:.1f} | Low >= {low_blur:.1f}")
    print(f"   -> SNR Thresholds:  Perfect >= {perf_snr:.1f} | Low >= {low_snr:.1f}\n")
    
    return perf_blur, low_blur, perf_snr, low_snr

def build_mixed_dataset(data_dir, output_folder, output_csv):
    """Randomly augments, mixes, and applies 3-tier labels based on baselines."""
    folders = {
        'perfect_output': 'perfect_light',
        'high_light_output': 'high_light',
        'low_light_output': 'low_light'
    }
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    perf_blur, low_blur, perf_snr, low_snr = get_baselines(os.path.join(data_dir, 'perfect_output'))
    
    processed_count = 0
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Class_Label', 'Brightness', 'Blurriness', 'SNR'])

        for folder_name, light_label in folders.items():
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.exists(folder_path):
                print(f"⚠️ Warning: Folder '{folder_name}' not found. Skipping.")
                continue
                
            print(f"\n⚙️ Processing {folder_name}...")
            
            all_items_in_folder = os.listdir(folder_path)
            total_items = len(all_items_in_folder)
            readable_images = 0
            
            for filename in all_items_in_folder:
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue

                readable_images += 1
                
                if random.choice([True, False]):
                    image = add_blur(image)
                if random.choice([True, False]):
                    image = add_noise(image)
                    
                b, bl, s = extract_features(image)
                
                if bl >= perf_blur:
                    blur_label = "perfect_blur"
                elif bl >= low_blur:
                    blur_label = "low_blur"
                else:
                    blur_label = "heavy_blur"

                if s >= perf_snr:
                    noise_label = "perfect_noise"
                elif s >= low_snr:
                    noise_label = "low_noise"
                else:
                    noise_label = "heavy_noise"
                
                final_class = f"{light_label}_{blur_label}_{noise_label}"
                
                new_filename = f"{light_label}_{processed_count}.jpg"
                cv2.imwrite(os.path.join(output_folder, new_filename), image)
                
                writer.writerow([new_filename, final_class, b, bl, s])
                processed_count += 1

            print(f"   📊 Folder Summary for '{folder_name}':")
            print(f"      - Total items found inside: {total_items}")
            print(f"      - Successfully read images: {readable_images}")
            if total_items != readable_images:
                print(f"      - ⚠️ Skipped {total_items - readable_images} items (wrong format or corrupted).")

    print(f"\n✅ All done! {processed_count} mixed images successfully saved to '{output_folder}'.")
    print(f"✅ CSV dataset successfully saved as '{output_csv}'.")

data_directory = "data/ori_file"
mixed_output_directory = "mixed_dataset" 
csv_output_file = "final_3tier_dataset.csv"

build_mixed_dataset(data_directory, mixed_output_directory, csv_output_file)