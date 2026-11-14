# Load volume on a passing forklift

Sentics challenge. Source: Linde E80 customer video, https://www.youtube.com/watch?v=RCqa8Kojlw8

## Result

| Corners | Width | Depth | Height | Volume | 90% range |
|---|---|---|---|---|---|
| Hand-marked | 2.40 m (known) | 2.50 m | 1.53 m | **9.1 m3** | 8.0 to 10.5 |
| Found automatically | 2.40 m (known) | 2.26 m | 1.47 m | **8.0 m3** | 6.8 to 9.7 |
| Counting check (2 x 3 EUR pallets, 5 layers) | 2.40 m | 2.40 m | 1.47 to 1.53 m | 8.5 to 8.8 m3 | |

Volume is the bounding box of the crates, pallets excluded. Add about 0.8 m3 for the pallets.
My answer: **between 8 and 9 m3.** The two ways of getting the corners disagree by about 1 m3,
and that gap is an honest measure of how sure I am.

The automatic corners land on average 15 px from the hand-marked ones (worst corner 41 px)
in a 1920 x 1080 frame.

## Which part of the video

The video has four forklift passes with different loads. I use the pass at 2:16 to 2:23
(`data/pass_0216.mp4`): the camera is static, the pallets are visible, and both the front and
the side of the load can be seen. It is treated as a camera feed and read frame by frame.

## How it works

1. `src/stream.py` reads frames from a file or an `rtsp://` URL through the same code path.
2. `src/detect.py` uses background subtraction (static camera) to notice the pass and pick the
   frames where the vehicle is largest, fully in view and sharp. It picks frames 236 to 256.
3. Six corners of the crate stack are needed on two of those frames. `src/autocorners.py` finds
   them without clicking:
   - Silhouette of everything that moved: difference to a background image (median of the clip).
   - Front face: the crate-blue region inside the silhouette. A line is fitted to the outer
     boundary of each side, the intersections are the front corners.
   - Bottom edge of the side: the sunlit pallet wood under the crates. The top of that strip is
     the edge, and where the strip stops is the far bottom corner.
   - Top edge of the side: hidden behind the mast, so its direction comes from the crate layer
     joints, which all aim at one vanishing point on the bottom edge line.

   The hand-marked corners (`annotations/corners.json`) are kept as the reference. In the UI you
   can switch between both and drag any corner.
4. `src/boxfit.py` fits one 3D box to both frames. The only real-world size given is the width:
   two EUR pallets side by side = 2.4 m. Focal length, camera pose, depth and height are solved.
5. The corners are shaken by their expected error (4 to 6 px hand-marked, 8 px automatic) and the
   box is refitted (200 times on the command line, 80 in the UI) to get the range.
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
2. **The six corners.** They start at the hand-marked positions. Press "Detect corners automatically"
   to let the program find them. Drag any yellow dot to change it.
3. **Result.** Press "Fit the box". The volume appears at once, the 90% range about 20 seconds later.
4. **Where it fails.** The limits of the method.

If port 8000 is busy, set another one first: `set PORT=8010` (Windows) or `PORT=8010 bash run.sh` (Linux / macOS).

Without the UI: `.venv/bin/python run.py` (Windows: `.venv\Scripts\python run.py`) writes
`out/result.json` (both results) and the overlay images. On a live camera, `run.py rtsp://camera/url` uses the same code path.

## The clip

`data/pass_0216.mp4` is seven seconds cut from the source video (2:16.4 to 2:23.3), included so the
app runs straight after cloning. The footage belongs to Linde Material Handling and is used here
only for this evaluation. To cut it yourself from the downloaded video:

    ffmpeg -ss 136.4 -i source.mp4 -t 6.9 -an -c:v libx264 -crf 16 data/pass_0216.mp4

## Scope

The app is built around this one clip. The box fit is general (any pass with a visible pallet
and a static camera). The automatic corner finder is not: it uses cues from this scene.
Run on nine different sets of frames from the same pass it gave 6.9 to 9.0 m3 eight times and
11.1 m3 once, and that one had a clearly worse fit error (17 px against 7 to 12), so it can be flagged.

## Where it fails

- Depth is the weakest number. The side face is seen at a steep angle, so a few pixels of corner error move depth by 10 cm or more.
- The automatic corners depend on this scene: blue crates, pallet wood lit by the sun, a load built from equal layers, and the layer count given (5). A different load or light needs other cues or a learned detector.
- The automatic result is about 1 m3 lower than the hand-marked one. Without ground truth I cannot say which is closer.
- The top and the back of the load are never seen. I assume a full rectangular stack.
- It needs a known-size object in view. No pallet, no scale.
- The background image needs the truck to keep moving. On a live camera it would come from the running background model.
- Lens distortion is ignored and the principal point is assumed at the image centre.

## On a fixed, calibrated camera

Focal length, floor plane and scale would be known in advance. Then one frame is enough,
no pallet is needed for scale, and the depth uncertainty drops sharply.
