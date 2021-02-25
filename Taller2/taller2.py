from imageShape import imageShape

if __name__ == '__main__':
    while True:
        try:
            w = int(input("Ingrese ancho de la imagen en píxeles: "))
            h = int(input("Ingrese alto de la imagen en píxeles, debe ser diferente del alto: "))
            if (w < 1 or h < 1) or (w == h):
                raise ValueError
            else:
                break
        except ValueError:
            print("Valores inválidos, intente de nuevo")

    img = imageShape(w, h)  # Crea objeto con parámetros indicados por el usuario
    img.generateShape()  # Genera imagen con figura geométrica
    img.showShape()  # Muestra imagen guardada en el objeto
    gen_img, gen_name = img.getShape()  # Obtiene el nombre de la figura generada
    what_name = img.whatShape(gen_img)  # Clasifica la figura generada anteriormente e indica el nombre

    # Verifica si la figura fue clasificada correctamente
    correct = (gen_name == what_name)
    print("\nFigura generada ->", gen_name, "\nClasificación ->", what_name)
    print("Figura clasificada correctamente" if correct else "Figura mal clasificada")