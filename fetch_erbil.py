import urllib.request
import math
import os
import time
import sys

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def download_tiles(lat, lon, zoom, grid_size=20, output_dir="erbil_raw_tiles"):
    os.makedirs(output_dir, exist_ok=True)
    cx, cy = deg2num(lat, lon, zoom)
    
    half_grid = grid_size // 2
    count = 0
    print(f"Downloading {grid_size}x{grid_size} tiles around Erbil ({lat}, {lon}) at zoom {zoom}...", flush=True)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for i in range(cx - half_grid, cx + half_grid):
        for j in range(cy - half_grid, cy + half_grid):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{j}/{i}"
            output_path = os.path.join(output_dir, f"tile_{i}_{j}.jpg")
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=5) as response, open(output_path, 'wb') as out_file:
                        out_file.write(response.read())
                    count += 1
                    if count % 10 == 0:
                        print(f"Downloaded {count} tiles...", flush=True)
                except Exception as e:
                    print(f"Failed to download tile {i},{j}: {e}", flush=True)
                time.sleep(0.05)
                
    print(f"Finished downloading new tiles.", flush=True)

if __name__ == "__main__":
    download_tiles(36.1901, 44.0090, zoom=17, grid_size=50) # 50x50 = 2500 tiles
