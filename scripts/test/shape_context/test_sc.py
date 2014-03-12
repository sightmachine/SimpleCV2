from simplecv import *

color = Color()
img = Image('badge.png')
img = img.invert()
blobs = img.find_blobs()
img2 = Image('deformed.png')
img2 = img2.invert()
blobs2 = img2.find_blobs()

for j in range(0,len(blobs)):
    data = blobs[j].ShapeContextMatch(blobs2[j])
    mapvals = data[0]
    fs1 = blobs[j].get_shape_context()
    fs1.draw()
    fs2 = blobs2[j].get_shape_context()
    fs2.draw()

img2 = img2.apply_layers()
img = img.apply_layers()
img3 = img.side_by_side(img2,'bottom')

for j in range(0,3):
    data = blobs[j].ShapeContextMatch(blobs2[j])
    mapvals = data[0]
    for i in range(0,len(blobs[j]._completeContour)):
    #img3.clear_layers()
        lhs = blobs[j]._completeContour[i]
        idx = mapvals[i];
        rhs = blobs2[j]._completeContour[idx[0]]
        rhsShift = (rhs[0],rhs[1]+img.height)
        img3.draw_line(lhs,rhsShift,color=color.getRandom(),thickness=1)
        img3.show()
