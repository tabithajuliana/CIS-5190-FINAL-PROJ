"""
prepare_data.py
────────────────
Walks your photos folder, extracts EXIF GPS data, infers time-of-day from
filenames, and writes metadata.csv ready for training.

Expected folder layouts (auto-detected):

  Layout A — flat folder with descriptive filenames:
      data/
          loc01_morning.jpg
          loc01_evening.jpg
          loc02_dawn.jpg
          ...

  Layout B — one subfolder per location, time as filename:
      data/
          loc01_locust_walk/
              morning.jpg
              evening.jpg
          loc02_college_hall/
              dawn.JPG
              night.JPG

  Layout C — one subfolder per location, descriptive filenames:
      data/
          loc01_locust_walk/
              morning_angle1.jpg
              morning_angle2.jpg
              evening_angle1.jpg

Usage:
    python src/prepare_data.py --data_dir data --output data/metadata.csv
"""

import os
import csv
import argparse
import re
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False
    print('Warning: pillow-heif not installed; .HEIC files will be skipped.')

from PIL.ExifTags import TAGS, GPSTAGS

TIME_LABELS = ['dawn', 'morning', 'noon', 'evening', 'night']
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.heic', '.JPG', '.JPEG', '.PNG', '.HEIC')


def extract_gps(image_path):
    """Pull GPS lat/lon from EXIF. Returns (lat, lon) or (None, None)."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None, None
        for tag_id, value in exif_data.items():
            if TAGS.get(tag_id) == 'GPSInfo':
                gps_info = {GPSTAGS.get(k): v for k, v in value.items()}
                lat = gps_info.get('GPSLatitude')
                lon = gps_info.get('GPSLongitude')
                lat_ref = gps_info.get('GPSLatitudeRef', 'N')
                lon_ref = gps_info.get('GPSLongitudeRef', 'W')
                if lat and lon:
                    lat_d = lat[0] + lat[1]/60 + lat[2]/3600
                    lon_d = lon[0] + lon[1]/60 + lon[2]/3600
                    if lat_ref == 'S': lat_d = -lat_d
                    if lon_ref == 'W': lon_d = -lon_d
                    return round(float(lat_d), 7), round(float(lon_d), 7)
    except Exception:
        pass
    return None, None


def infer_time_from_name(filename):
    """Extract time-of-day word from a filename. Case-insensitive."""
    name = os.path.splitext(filename)[0].lower()
    for t in TIME_LABELS:
        if re.search(rf'\b{t}\b', name) or t in name:
            return t
    return None


def collect_photos(data_dir):
    """Walk data_dir, infer location_id and time for each image."""
    rows = []
    has_subfolders = any(
        os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
        for d in os.listdir(data_dir)
    )

    if has_subfolders:
        # Layouts B and C — subfolder = location
        for loc_id in sorted(os.listdir(data_dir)):
            loc_path = os.path.join(data_dir, loc_id)
            if not os.path.isdir(loc_path) or loc_id.startswith('.'):
                continue
            for fname in sorted(os.listdir(loc_path)):
                if not fname.lower().endswith(IMG_EXTS) or fname.startswith('.'):
                    continue
                full_path = os.path.join(loc_path, fname)
                rel_path = os.path.join(loc_id, fname)
                time = infer_time_from_name(fname)
                lat, lon = extract_gps(full_path)
                rows.append({
                    'filename':    rel_path,
                    'location_id': loc_id,
                    'time':        time or '',
                    'lat':         lat or '',
                    'lon':         lon or '',
                })
    else:
        # Layout A — flat folder, location_id from filename prefix
        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith(IMG_EXTS) or fname.startswith('.'):
                continue
            full_path = os.path.join(data_dir, fname)
            time = infer_time_from_name(fname)
            # Try to extract location prefix: 'loc01_morning.jpg' -> 'loc01'
            base = os.path.splitext(fname)[0].lower()
            for t in TIME_LABELS:
                base = base.replace(f'_{t}', '').replace(t, '')
            loc_id = re.sub(r'[^a-z0-9_]', '', base).strip('_') or 'unknown'
            lat, lon = extract_gps(full_path)
            rows.append({
                'filename':    fname,
                'location_id': loc_id,
                'time':        time or '',
                'lat':         lat or '',
                'lon':         lon or '',
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Folder with images')
    parser.add_argument('--output',   default='metadata.csv')
    args = parser.parse_args()

    rows = collect_photos(args.data_dir)

    if not rows:
        print(f'No images found in {args.data_dir}')
        return

    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'location_id', 'time', 'lat', 'lon'])
        w.writeheader()
        w.writerows(rows)

    print(f'\nWrote {len(rows)} entries -> {args.output}')

    # Summary
    by_time     = {t: 0 for t in TIME_LABELS}
    by_loc      = {}
    missing_time = 0
    missing_gps  = 0
    for r in rows:
        if r['time'] in by_time:
            by_time[r['time']] += 1
        else:
            missing_time += 1
        by_loc[r['location_id']] = by_loc.get(r['location_id'], 0) + 1
        if not r['lat']:
            missing_gps += 1

    print(f'\nLocations: {len(by_loc)}')
    print(f'Time-of-day distribution: {by_time}')
    print(f'Missing time labels: {missing_time}')
    print(f'Missing GPS data:    {missing_gps}')

    if missing_time > 0:
        print(f'\nNote: {missing_time} images have no time label inferred from filename.')
        print('Open metadata.csv and fill in the empty "time" column manually.')


if __name__ == '__main__':
    main()
