import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk


class BoundingBoxApp:
    def __init__(self, container):
        self.container = container
        self.container.title("Bounding Box Annotator")

        # Create canvas for image display
        self.frame = tk.Frame(self.container, width=500, height=500)
        self.frame.pack(expand=True, fill=tk.BOTH) #.grid(row=0,column=0)
        self.canvas = tk.Canvas(self.frame, width=500, height=500, scrollregion=(0,0,500,500))
        hbar = tk.Scrollbar(self.frame, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM,fill=tk.X)
        hbar.config(command=self.canvas.xview)
        vbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT,fill=tk.Y)
        vbar.config(command=self.canvas.yview)
        self.canvas.config(width=300, height=300)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack(side = tk.LEFT, expand = True, fill = tk.BOTH)

        # Button to load image
        self.load_button = tk.Button(container, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Attach event listeners
        self.polygon = None
        self.points = []
        self.canvas.tag_bind('all', "<Button-1>", self.draw_bbox)

    def draw_bbox(self, event):
        self.points.append((event.x, event.y))
        if 2 < len(self.points):
            if self.polygon:
                print(dir(self.polygon))
            geom = []
            for point in self.points:
                geom += list(point)
            self.polygon = self.canvas.create_polygon(geom, outline='blue', fill=None, width=2)
            #rectangle = canvas.create_rectangle(50, 50, 250, 150, fill="blue")
        

    def resize_canvas(self, width=500, height=500):
        self.canvas.height = height
        self.canvas.width = width
        self.canvas.config(
            width=self.canvas.width, 
            height=self.canvas.height,
            scrollregion=(0,0,self.canvas.width,self.canvas.height)
        )

    def load_image(self):
        # Load image and display on canvas
        filepath = filedialog.askopenfilename()
        if filepath:
            self.image = cv2.imread(filepath)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(self.image))
            self.resize_canvas(
                height = self.photo.height(), 
                width = self.photo.width()
            )
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

root = tk.Tk()
app = BoundingBoxApp(root)
root.mainloop()


# https://docs.ultralytics.com/datasets/segment/
# <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>