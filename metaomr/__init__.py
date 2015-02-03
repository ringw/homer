def open(image_path):
    # Lazy import allows us to import submodules without initializing the GPU
    from . import image, page
    images = image.read_pages(image_path)
    return map(page.Page, images)
