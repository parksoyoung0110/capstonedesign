from io import BytesIO
from os import path
import requests
from PIL import Image

if __name__ == "__main__":
    file_root = path.join(path.abspath(path.dirname(__file__)), "images")

    # The test images are downloaded from
    # http://www.r0k.us/graphics/kodak/
    # and are cleared for unrestricted usage
    url_root = "http://www.r0k.us/graphics/kodak/kodak/"

    # Only the portrait images are used to avoid cropping before the comparison
    portrait_images_idcs = (4, 9, 10, 17, 18, 19)
    for idx in portrait_images_idcs:
        file_name = f"kodim{idx:02d}.png"

        file = path.join(file_root, file_name)
        url = url_root + file_name

        print(f"Downloading content of {url} to {file}")
        image = Image.open(BytesIO(requests.get(url).content))
        image = image.convert("L")
        image.save(file)
