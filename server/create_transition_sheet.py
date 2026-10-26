
import cv2
from pathlib import Path

input_dir = Path("transition_inspection")
output_path = input_dir / "transition_contact_sheet.jpg"

files = sorted(input_dir.glob("frame_*.jpg"))

if not files:
    raise FileNotFoundError(f"No frames found in {input_dir.resolve()}")

tiles = []

for path in files:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Could not read {path}")
        continue

    img = cv2.resize(img, (424, 239))

    label = path.stem.replace("frame_", "Frame ")
    cv2.rectangle(img, (0, 0), (160, 32), (0, 0, 0), -1)
    cv2.putText(
        img, label, (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2
    )
    tiles.append(img)

if not tiles:
    raise RuntimeError("No readable frames found.")

columns = 2
rows = []

for i in range(0, len(tiles), columns):
    row = tiles[i:i + columns]

    if len(row) == 1:
        row.append(tiles[-1].copy())

    rows.append(cv2.hconcat(row))

sheet = cv2.vconcat(rows)

if cv2.imwrite(str(output_path), sheet):
    print(f"SUCCESS: {output_path.resolve()}")
else:
    raise RuntimeError("Failed to save contact sheet.")

print(f"Frames included: {len(files)}")
