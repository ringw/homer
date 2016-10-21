from homer.page import Page
from homer.rotate import get_rotated_page
from homer.scale import get_scaled_page

def create_page(image):
  page = Page(image)
  page = get_rotated_page(page)
  page = get_scaled_page(page)
  return page
