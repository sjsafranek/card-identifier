import cv2


class Rectangle(object):

    def __init__(self, x=None, y=None, w=None, h=None, contour=None):
        if None is not contour:
            x, y, w, h = cv2.boundingRect(contour)
        self.x = x 
        self.y = y
        self.w = w
        self.h = h

    @property
    def topLeft(self):
        return (self.minX, self.minY)

    @property
    def bottomRight(self):
        return (self.maxX, self.maxY)

    @property
    def minX(self):
        return self.x

    @property
    def maxX(self):
        return self.x + self.w

    @property
    def minY(self):
        return self.y

    @property
    def maxY(self):
        return self.y + self.h

