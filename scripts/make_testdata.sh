#!/usr/bin/env bash
# Regenerate the Stage 2 test clips from kiarina/labs shared assets.
# Usage: scripts/make_testdata.sh /path/to/labs/tests/assets/jpg
set -euo pipefail
A=${1:?usage: make_testdata.sh <labs tests/assets/jpg>}
mkdir -p testdata

ffmpeg -loglevel error -y -loop 1 -i "$A/objects_1536x1024_358kb.jpg" \
  -vf "zoompan=z='min(zoom+0.0015,1.4)':d=120:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=768x512,fps=30" \
  -t 4 -c:v libx264 -pix_fmt yuv420p -g 30 testdata/pan_objects.mp4

ffmpeg -loglevel error -y \
  -loop 1 -t 2 -i "$A/street_scene_1774x887_287kb.jpg" \
  -loop 1 -t 2 -i "$A/ocr_1448x1086_242kb.jpg" \
  -filter_complex "[0:v]scale=768:512,setsar=1,fps=30[a];[1:v]scale=768:512,setsar=1,fps=30[b];[a][b]concat=n=2:v=1:a=0" \
  -c:v libx264 -pix_fmt yuv420p -g 30 testdata/street_ocr.mp4

# Deliberately non-32-aligned (700x460) to exercise the resize path.
ffmpeg -loglevel error -y -loop 1 -i "$A/many_face_1280x720_275kb.jpg" \
  -vf "zoompan=z='min(zoom+0.001,1.25)':d=90:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=700x460,setsar=1,fps=30" \
  -t 3 -c:v libx264 -pix_fmt yuv420p -g 30 testdata/faces_odd.mp4
