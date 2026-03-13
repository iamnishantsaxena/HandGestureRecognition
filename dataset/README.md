# Dataset Directory

This directory should contain the HaGRID (Hand Gesture Recognition Image Dataset).

## Download Instructions

1. Visit the [HaGRID GitHub repository](https://github.com/hukenovs/hagrid)
2. Download the dataset (approximately 50GB)
3. Extract the dataset into this `dataset/` directory
4. Ensure the structure matches:
   ```
   dataset/
   ├── ann_subsample.json          # Combined annotations (run combine_json.py if needed)
   ├── call/                       # Gesture class folders
   ├── dislike/
   ├── fist/
   ├── four/
   ├── like/
   ├── mute/
   ├── ok/
   ├── one/
   ├── palm/
   ├── peace/
   ├── peace_inverted/
   ├── rock/
   ├── stop/
   ├── stop_inverted/
   ├── test/                       # Test images
   ├── three/
   ├── three2/
   ├── two_up/
   └── two_up_inverted/
   ```

## Annotation Processing

If you have individual JSON annotation files, run:
```bash
python utils/combine_json.py
```
This will combine them into `ann_subsample.json`.

## Dataset Details

- **Total Images**: 552,992 FullHD RGB images
- **Classes**: 18 gestures + 1 "no_gesture" class
- **Resolution**: 1920×1080 (processed to 224×224 for training)
- **Split**: 92% training, 8% testing (by user_id)