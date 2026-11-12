# Load volume on a passing forklift

Sentics challenge. Source: Linde E80 customer video, https://www.youtube.com/watch?v=RCqa8Kojlw8

## Result

| | Width | Depth | Height | Volume |
|---|---|---|---|---|
| Box fit (measured) | 2.40 m (known) | 2.50 m | 1.53 m | **9.1 m3** |
| 90% range | | 2.33 to 2.71 m | 1.47 to 1.59 m | 8.0 to 10.5 m3 |
| Counting check (2 x 3 EUR pallets, 5 layers) | 2.40 m | 2.40 m | 1.53 m | 8.8 m3 |

Volume is the bounding box of the crates, pallets excluded. Including the pallets it is about 9.6 m3.
My answer: **about 9 m3, plus or minus 1.**

## Which part of the video

The video has four forklift passes with different loads. I use the pass at 2:16 to 2:23
(`data/pass_0216.mp4`): the camera is static, the pallets are visible, and both the front and
the side of the load can be seen. It is treated as a camera feed and read frame by frame.

## How it works

1. `src/stream.py` reads frames from a file or an `rtsp://` URL through the same code path.
2. `src/detect.py` uses background subtraction (static camera) to notice the pass and pick the
   frames where the vehicle is largest, fully in view and sharp. It picks frames 236 to 256.
3. Six corners of the crate stack are marked on two of those frames (`annotations/corners.json`).
   This step is manual today. In the UI the corners can be dragged and the box refitted.
4. `src/boxfit.py` fits one 3D box to both frames. The only real-world size given is the width:
   two EUR pallets side by side = 2.4 m. Focal length, camera pose, depth and height are solved.
5. The clicked corners are shaken by their click error and the box is refitted (200 times on the
   command line, 80 in the UI) to get the range.
6. Cross-check: the fitted depth (2.5 m) rounds to 3 pallets of 0.8 m, so the footprint is 2.4 x 2.4 m.

Plausibility: 6 pallets x 40 crates = 240 crates, roughly 4 t, which fits an 8 t truck.
Fitted layer height is 0.305 m per crate.

## Run it

Needs Python 3.9 or newer and git. Nothing else to install by hand.

    git clone https://github.com/pr-dev-ai/forklift-load-volume
    cd forklift-load-volume

| System | Command |
|---|---|
| Windows | double-click `run.bat`, or type `run.bat` in a terminal |
| Linux / macOS | `bash run.sh` |

The script creates a virtual environment in `.venv`, installs the two requirements
(opencv and numpy) and starts the app. The first run takes a minute or two.
The browser then opens http://localhost:8000. Stop the app with Ctrl+C.

In the browser:

1. **The feed.** Press "Detect the pass" to see when the truck is in view and which frames were picked.
2. **Mark the six corners.** The corners are pre-marked on two frames. Drag any yellow dot to change it.
3. **Result.** Press "Fit the box". The volume appears at once, the 90% range about 20 seconds later.
4. **Where it fails.** The limits of the method.

If port 8000 is busy, set another one first: `set PORT=8010` (Windows) or `PORT=8010 bash run.sh` (Linux / macOS).

Without the UI: `.venv/bin/python run.py` (Windows: `.venv\Scripts\python run.py`) writes
`out/result.json` and the overlay images. On a live camera, `run.py rtsp://camera/url` uses the same code path.

## The clip

`data/pass_0216.mp4` is seven seconds cut from the source video (2:16.4 to 2:23.3), included so the
app runs straight after cloning. The footage belongs to Linde Material Handling and is used here
only for this evaluation. To cut it yourself from the downloaded video:

    ffmpeg -ss 136.4 -i source.mp4 -t 6.9 -an -c:v libx264 -crf 16 data/pass_0216.mp4

## Scope

The app is built around this one clip. The method is general (any forklift pass with a visible
pallet and a static camera), but a new video needs two manual inputs that are fixed here:
the six corners (`annotations/corners.json`) and the number of pallets across the front
(`PALLETS_ACROSS` in `src/measure.py`). Choosing a video and marking corners in the UI is the next step.

## Where it fails

- Depth is the weakest number. The side face is seen at a steep angle, so a few pixels of click error move depth by 10 cm or more.
- The top and the back of the load are never seen. I assume a full rectangular stack.
- It needs a known-size object in view. No pallet, no scale.
- Corners are hand-marked. Automatic corner finding is not built yet; strong backlight in this clip makes it hard.
- The pass detector assumes a static camera. It does not work on the other scenes of this video.
- Lens distortion is ignored and the principal point is assumed at the image centre.

## On a fixed, calibrated camera

Focal length, floor plane and scale would be known in advance. Then one frame is enough,
no pallet is needed for scale, and the depth uncertainty drops sharply.
