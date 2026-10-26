
import cv2
from pathlib import Path

INPUT_DIR = Path("transition_inspection")
OUTPUT_PATH = INPUT_DIR / "transition_contact_sheet.jpg"

frame_files = sorted(INPUT_DIR.glob("frame_*.jpg"))

if not frame_files:
    raise FileNotFoundError(
        f"No extracted frames found in {INPUT_DIR.resolve()}"
    )

tiles = []

for file_path in frame_files:
    image = cv2.imread(str(file_path))

    if image is None:
        print(f"Skipping unreadable image: {file_path}")
        continue

    # Resize for a compact contact sheet.
    image = cv2.resize(image, (424, 239))

    label = file_path.stem.replace("frame_", "Frame ")

    cv2.rectangle(image, (0, 0), (150, 30), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    tiles.append(image)

if not tiles:
    raise RuntimeError("No readable frames were found.")

columns = 2
rows = (len(tiles) + columns - 1) // columns

sheet = cv2.copyMakeBorder(
    cv2.vconcat([
        cv2.hconcat(
            tiles[row * columns:(row + 1) * columns]
            + [
                tiles[row * columns]
            ] * (columns - len(tiles[row * columns:(row + 1) * columns]))
        )
        for row in range(rows)
    ]),
    10, 10, 10, 10,
    cv2.BORDER_CONSTANT,
    value=(40, 40, 40)
)

if cv2.imwrite(str(OUTPUT_PATH), sheet):
    print(f"Contact sheet saved: {OUTPUT_PATH.resolve()}")
else:
    raise RuntimeError("Could not save contact sheet.")
