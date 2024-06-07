from imageShape import imageShape

if __name__ == '__main__':
    while True:
        try:
            w = int(input("Type width of the image in pixels: "))
            h = int(input("Type height of the image in pixels, it must be different from the width: "))
            if (w < 1 or h < 1) or (w == h):
                raise ValueError
            else:
                break
        except ValueError:
            print("Values are not valid, try again...")

    img = imageShape(w, h)  # Creates object with values specified by user
    img.generateShape()  # Generates image with geometric shape
    img.showShape()  # Shows image saved in object
    gen_img, gen_name = img.getShape()
    what_name = img.whatShape(gen_img)  # Gets the name of the generated figure

    # Verifies if figured was classified correctly
    correct = (gen_name == what_name)
    print("\nGenerated figure ->", gen_name, "\nClassificaiton ->", what_name)
    print("Figure was classified correctly" if correct else "Figure was misclassified.")