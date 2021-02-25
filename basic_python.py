'''1. Cree un programa que le pida al usuario que
ingrese su nombre y su edad. Imprima un mensaje
dirigido a ellos que les indique el año en que
cumplirán 100 años.'''
# import datetime

# name = input('Nombre: ')

# while True:
#     try:
#         age = int(input('Edad: '))
#         break
#     except:
#         print('Valor inválido, intente otra vez')

# bday = datetime.date.today().year - age
# year = bday + 100

# print(name + ' cumpliras 100 años en ', year)

'''2. Tome una lista y escriba un programa que 
imprima todos los elementos de la lista que sean 
inferiores a 10.'''
a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
b = [item for item in a if item <= 10]
print(b)
'''3. Tome dos listas y escriba un programa que
devuelva una lista que contenga solo los elementos
que son comunes entre las listas (sin duplicados). 
Asegúrese de que su programa funcione en dos listas
de diferentes tamaños.'''
# a = [1, 2, 2, 4, 5, 8, 11, 20, 34, 55, 89]
# b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# a_nodup = list(dict.fromkeys(a))
# b_nodup = list(dict.fromkeys(b))

# common = [item for item in a_nodup if item in b_nodup]
# print("Elementos comunes", common)

'''4. Modifique el programa del punto 3 para que genere 
al azar las dos listas.'''
# from random import randint
# a = [randint(0, 100) for i in range(randint(2, 10))]
# b = [randint(0, 100) for i in range(randint(2, 10))]
# print('a: ', a)
# print('b: ', b)

# common = list(set(a).intersection(b))
# print("Elementos comunes: ", common)
