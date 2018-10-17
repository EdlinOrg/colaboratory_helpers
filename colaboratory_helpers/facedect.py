#import sys

import dlib

# filename="face.jpg"

# https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/extractSubImage.m


def getFaces(filename, margin=0.2, verbose=False, extractPartial=False):

    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(filename)

    (imgHeight, imgWidth, _) = img.shape

    dets = detector(img, 1)

    if verbose:
        print("Number of faces detected: {}".format(len(dets)))

    faces=[]

    for k, d in enumerate(dets):
        if verbose:
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))

        width = d.right() - d.left()
        height = d.bottom() - d.top()
        if verbose:
            print("width {} height {}".format(width, height))

        sideMargin = int(width * margin)
        heightMargin = int(height * margin)

        newTop = d.top() - heightMargin

        if newTop < 0:
            newTop = 0
            if verbose:
                print("newTop below 0")
            if not extractPartial:
                faces.append(False)
                continue #lets skip it

        newBottom = d.bottom() + heightMargin

        if newBottom > imgHeight-1:
            newBottom = imgHeight-1
            if verbose:
                print("newBottom more than height")
            if not extractPartial:
                faces.append(False)
                continue #lets skip it

        newLeft = d.left() - sideMargin

        if newLeft < 0:
            newLeft = 0
            if verbose:
                print("newleft below 0")
            if not extractPartial:
                faces.append(False)
                continue #lets skip it

        newRight = d.right() + sideMargin


        if newRight > imgWidth -1:
            newRight = imgWidth -1
            if verbose:
                print("newRight more than width")
            if not extractPartial:
                faces.append(False)
                continue #lets skip it

        if verbose:
            print("New values: Left: {} Top: {} Right: {} Bottom: {}".format(
                newLeft, newTop, newRight, newBottom))

        # crop_img = img[d.top():d.bottom(),d.left():d.right()]
        crop_img = img[newTop:newBottom, newLeft:newRight]
        faces.append(crop_img)
#        plt.imshow(crop_img);

    return faces
