import sys
import image
import staff
import glyph
import numpy as np
import os
from PyQt4.QtGui import *
from PyQt4.QtCore import *

glyphs = []
glyph_assignments = []

class MainWindow(QWidget):
  def __init__(self):
    super(MainWindow, self).__init__()
    self.initUI()

  def initUI(self):
    self.imLabel = QLabel('hi')
    self.listView = QListView()
    self.listViewModel = GlyphListModel(self)
    self.listView.setModel(self.listViewModel)

    hbox = QHBoxLayout()
    hbox.addWidget(self.imLabel)
    hbox.addWidget(self.listView)

    vbox = QVBoxLayout()
    self.numLabel = QLabel('glyph #0')
    vbox.addWidget(self.numLabel)
    
    vbox.addLayout(hbox)
    self.saveButton = QPushButton('Save', self)
    self.saveButton.clicked.connect(self.saveButtonClicked)
    vbox.addWidget(self.saveButton)

    self.setLayout(vbox)

    self.setGeometry(300,300,300,150)
    self.setWindowTitle('Classifier')
    self.glyph_num = 0
    pix = QPixmap.fromImage(glyphs[0])
    scaled = pix.scaled(self.imLabel.size(), Qt.KeepAspectRatio)
    self.imLabel.setPixmap(scaled)
    self.show()

  def keyPressEvent(self, event):
    if type(event) == QKeyEvent and event.key() == 16777220: # return
      event.accept()
      indexes = self.listView.selectionModel().selectedIndexes()
      if len(indexes):
        ind = indexes[0].row()
        glyph_type = self.listViewModel.glyph_types[ind]
        print glyph_type
        glyph_assignments.append(glyph_type)
        if len(glyph_assignments) < len(glyphs):
          pix = QPixmap.fromImage(glyphs[len(glyph_assignments)])
          scaled = pix.scaled(self.imLabel.size(), Qt.KeepAspectRatio)
          self.imLabel.setPixmap(scaled)
          self.numLabel.setText('glyph #' + str(len(glyph_assignments)))
    else:
      event.ignore()

  def saveButtonClicked(self):
    os.makedirs('classifier')
    open_files = dict()
    global glyphsTask
    for ind, assignment in enumerate(glyph_assignments):
      if not (assignment in open_files):
        open_files[assignment] = open('classifier/' + assignment, 'a')
      open_files[assignment].write(','.join(map(str, glyphsTask.hu_moments[ind])) + '\n')

class GlyphListModel(QAbstractListModel):
  def __init__(self, parent=None, *args):
    QAbstractListModel.__init__(self, parent, *args)
    attr_val = dict()
    for attr in dir(glyph.Glyph):
      val = getattr(glyph.Glyph, attr)
      if type(val) is int:
        attr_val[attr] = val
    self.glyph_types = sorted(attr_val, key=attr_val.get)
  def rowCount(self, parent=QModelIndex()):
    return len(self.glyph_types)
  def data(self, index, role):
    if index.isValid() and role == Qt.DisplayRole:
      return QVariant(self.glyph_types[index.row()])
    else:
      return QVariant()

def main():
  app = QApplication(sys.argv)

  page, = image.read_pages(sys.argv[1])
  staff.StavesTask(page).process()
  global glyphsTask
  glyphsTask = glyph.GlyphsTask(page)
  glyphsTask.process()
  # Generate glyph bounding boxes
  bounds = np.zeros((glyphsTask.num_glyphs + 1, 4), dtype=int)
  bounds[:,0] = page.im.shape[1]
  bounds[:,2] = page.im.shape[0]
  for i, row in enumerate(glyphsTask.labelled_glyphs):
    in_row = np.unique(row)
    in_row_bool = np.in1d(np.arange(bounds.shape[0]), in_row)
    bounds[(bounds[:,0] > i) & in_row_bool, 0] = i
    bounds[(bounds[:,1] < i) & in_row_bool, 1] = i
  for j, col in enumerate(glyphsTask.labelled_glyphs.T):
    in_col = np.unique(col)
    in_col_bool = np.in1d(np.arange(bounds.shape[0]), in_col)
    bounds[(bounds[:,2] > j) & in_col_bool, 2] = j
    bounds[(bounds[:,3] < j) & in_col_bool, 3] = j
  for i in xrange(glyphsTask.num_glyphs):
    y_min,y_max,x_min,x_max = bounds[i+1]
    left = max(0, x_min - 100)
    right = min(page.im.shape[1], x_max + 100)
    top = max(0, y_min - 100)
    bottom = min(page.im.shape[0], y_max + 100)
    total = np.zeros((bottom - top, right - left, 4), np.int8)
    total[..., 3] = 255
    im_surrounding = page.im[top:bottom, left:right]
    total[im_surrounding.nonzero() + (slice(0,3),)] = 255
    # Color actual glyph red
    glyph_box = glyphsTask.labelled_glyphs[y_min:y_max+1, x_min:x_max+1]
    glyph_y, glyph_x = np.array(np.where(glyph_box == (i+1)))
    glyph_y += y_min - top
    glyph_x += x_min - left
    total[glyph_y, glyph_x, 0:2] = 0
    glyphs.append(QImage(total, total.shape[1], total.shape[0],
                  QImage.Format_RGB32))
    #glyphs.append(page.im[slice(y_min, y_max+1), slice(x_min, x_max+1)])
  mw = MainWindow()

  sys.exit(app.exec_())

if __name__ == '__main__':
  main()
