from . import image, page

def open(image_path):
    images = image.read_pages(image_path)
    return map(page.Page, images)
