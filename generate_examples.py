"""Create synthetic layout images for trying the scene indexer."""
from pathlib import Path
from PIL import Image, ImageDraw

directory = Path(__file__).resolve().parent / "examples" / "frames"
directory.mkdir(parents=True, exist_ok=True)
for name, offset in [("layout-a", 0), ("layout-a-near", 3), ("layout-b", 90)]:
    image = Image.new("RGB", (400, 300), "#146858")
    draw = ImageDraw.Draw(image)
    draw.ellipse((45 + offset, 85, 135 + offset, 175), fill="#edb143")
    draw.rectangle((230 - offset, 160, 300 - offset, 220), fill="#dce4ef")
    image.save(directory / (name + ".jpg"))
print(directory)
