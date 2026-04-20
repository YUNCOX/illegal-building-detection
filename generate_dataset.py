import os
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm

def create_complex_building(img, x, y, size):
    # Generates L-shape, U-shape or square building
    shape_type = random.choice(['square', 'L_shape', 'U_shape'])
    
    # Erbil often has beige/sandy or flat white concrete roofs
    color = (
        random.randint(180, 240), # B
        random.randint(180, 240), # G
        random.randint(180, 240)  # R
    )
    
    if shape_type == 'square':
        w, h = random.randint(15, size), random.randint(15, size)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (50, 50, 50), 1)
        return (x, y, x+w, y+h)
    
    elif shape_type == 'L_shape':
        w1, h1 = random.randint(20, size), random.randint(10, 20)
        w2, h2 = random.randint(10, 20), random.randint(20, size)
        # Main wing
        cv2.rectangle(img, (x, y), (x+w1, y+h1), color, -1)
        cv2.rectangle(img, (x, y), (x+w1, y+h1), (50, 50, 50), 1)
        # Side wing
        cv2.rectangle(img, (x, y), (x+w2, y+h2), color, -1)
        cv2.rectangle(img, (x, y), (x+w2, y+h2), (50, 50, 50), 1)
        return (x, y, x+max(w1, w2), y+max(h1, h2))
        
    elif shape_type == 'U_shape':
        w, h = random.randint(25, size), random.randint(25, size)
        thick = random.randint(8, 12)
        cv2.rectangle(img, (x, y), (x+w, y+thick), color, -1) # top
        cv2.rectangle(img, (x, y), (x+thick, y+h), color, -1) # left
        cv2.rectangle(img, (x+w-thick, y), (x+w, y+h), color, -1) # right
        
        cv2.rectangle(img, (x, y), (x+w, y+thick), (50, 50, 50), 1)
        cv2.rectangle(img, (x, y), (x+thick, y+h), (50, 50, 50), 1)
        cv2.rectangle(img, (x+w-thick, y), (x+w, y+h), (50, 50, 50), 1)
        return (x, y, x+w, y+h)

def add_shadows(img, mask_rect, offset=4):
    x1, y1, x2, y2 = mask_rect
    # Darken pixels below and to the right to simulate sunlight shadow
    shadow_x1 = x1 + offset
    shadow_y1 = y1 + offset
    shadow_x2 = x2 + offset
    shadow_y2 = y2 + offset
    
    # ensure bounds
    shadow_x1 = max(0, min(shadow_x1, 255))
    shadow_y1 = max(0, min(shadow_y1, 255))
    shadow_x2 = max(0, min(shadow_x2, 256))
    shadow_y2 = max(0, min(shadow_y2, 256))
    
    if shadow_x2 > shadow_x1 and shadow_y2 > shadow_y1:
        roi = img[shadow_y1:shadow_y2, shadow_x1:shadow_x2]
        img[shadow_y1:shadow_y2, shadow_x1:shadow_x2] = (roi * 0.4).astype(np.uint8)

def augment_image(img):
    # Random brightness/contrast to simulate different satellite passes
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-15, 15)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Random flip to get more variations from the same tile
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.5:
        img = cv2.flip(img, 0)
        
    return img

def generate_semi_synthetic_dataset(tiles_dir="erbil_raw_tiles", output_dir="dataset", variants_per_tile=3):
    tiles = glob.glob(os.path.join(tiles_dir, "*.jpg"))
    if not tiles:
        print("No raw tiles found! Run fetch_erbil.py first.")
        return
        
    print(f"Found {len(tiles)} real Erbil tiles. Generating semi-synthetic dataset...")
    
    dirs = ['baseline', 'recent', 'mask']
    splits = ['train', 'val']
    
    for split in splits:
        for d in dirs:
            os.makedirs(os.path.join(output_dir, split, d), exist_ok=True)
            
    sample_idx = 0
    for tile_path in tqdm(tiles):
        original_tile = cv2.imread(tile_path)
        if original_tile is None or original_tile.shape != (256, 256, 3):
            continue
            
        # Create 5 variations out of every 1 real Erbil background
        for v in range(variants_per_tile):
            baseline = augment_image(original_tile.copy())
            recent = baseline.copy()
            mask = np.zeros((256, 256), dtype=np.uint8)
            
            # Inject new "illegal" buildings into the recent image
            for _ in range(random.randint(1, 4)):
                x, y = random.randint(10, 200), random.randint(10, 200)
                rect = create_complex_building(recent, x, y, size=random.randint(30, 50))
                add_shadows(recent, rect)
                
                rx1, ry1, rx2, ry2 = rect
                cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
                
            split = 'train' if random.random() < 0.85 else 'val'
            
            cv2.imwrite(os.path.join(output_dir, split, 'baseline', f"{sample_idx}.png"), baseline)
            cv2.imwrite(os.path.join(output_dir, split, 'recent', f"{sample_idx}.png"), recent)
            cv2.imwrite(os.path.join(output_dir, split, 'mask', f"{sample_idx}.png"), mask)
            
            sample_idx += 1
            
    print(f"Generated {sample_idx} HIGH-QUALITY Erbil image pairs successfully!")

if __name__ == "__main__":
    generate_semi_synthetic_dataset()
