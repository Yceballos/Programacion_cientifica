#!/usr/bin/python3.5
#encoding: latin-1
from multiprocessing import Pool as mp
from multiprocessing import cpu_count
import pandas as pd
import os
import time
import sys
import multiprocessing as mp


def process_line(line):
    # Cuenta la frecuencia de cada carácter
    counter = {}
    for letter in line:
        if letter not in counter:
            counter[letter] = 0
        counter[letter] += 1

    # Encuentra el caracter con la mayor frecuencia usando `counter`
    most_frequent_letter = None
    max_count = 0
    for key, value in counter.items():
        if value >= max_count:
            max_count = value
            most_frequent_letter = key
    return most_frequent_letter

#################################################Programación serial#####################################################################################
# Lectura del archivo, se debe indicador el encoding
def serial_read(file_name):
    results = []
    with open(file_name, encoding='latin-1') as f:
        for line in f:
            results.append(process_line(line))
    return results

##################################################Progranación paralelo#################################################################################

def parallel_read(file_name):
    # Máximo número de procesos que se pueden ejecutar
    #cpu_count = mp.cpu_count()
    cpu_count = file_name[1]
    file_name=file_name[0]


    print ("Paralelizacion con:",cpu_count)

    file_size = os.path.getsize(file_name)
    chunk_size = file_size // cpu_count

    # Arguments para cada chunk (eg. [('input.txt', 0, 32), ('input.txt', 32, 64)])
    chunk_args = []
    with open(file_name, encoding='latin-1') as f:

        def is_start_of_line(position):
            if position == 0:
                return True
            # Verificar si es último carácter y devuelve la última posición
            f.seek(position - 1)
            return f.read(1) == '\n'

        def get_next_line_position(position):
            # Lectura de la línea actual
            f.seek(position)
            f.readline()
            # Retorna la posición luego de leer la línea
            return f.tell()

        chunk_start = 0
        # Iterar todos los chunks y construir argumentos para la paralelización `process_chunk`
        while chunk_start < file_size:
            chunk_end = min(file_size, chunk_start + chunk_size)

            # Verificar que el chunck termina en el inicio de otra palabra
            while not is_start_of_line(chunk_end):
                chunk_end -= 1

            if chunk_start == chunk_end:
                chunk_end = get_next_line_position(chunk_end)

            # Guardar argumentos de `process_chunk`
            args = (file_name, chunk_start, chunk_end)
            chunk_args.append(args)

            # Moverse al siguiente chunk
            chunk_start = chunk_end

    with mp.Pool(cpu_count) as p:
        # Correr en paralelo
        chunk_results = p.starmap(process_chunk, chunk_args)

    results = []
    # Combina los resultados de cada chunk `results`
    for chunk_result in chunk_results:
        for result in chunk_result:
            results.append(result)
    return results


def process_chunk(file_name, chunk_start, chunk_end):
    chunk_results = []
    with open(file_name, encoding='latin-1') as f:
        # Mover la pos a `chunk_start`
        f.seek(chunk_start)

        # Leer las líneas hasta `chunk_end`
        for line in f:
            chunk_start += len(line)
            if chunk_start > chunk_end:
                break
            chunk_results.append(process_line(line))
    return chunk_results


def measure(func, *args):
    time_start = time.time()
    result = func(*args)
    timu=28
    time_end = time.time()
    print(func,time_end - time_start)
    return result


if __name__ == '__main__':
    print(str(sys.argv[1])," - ",sys.argv[2])
    measure(serial_read, str(sys.argv[1]))
    measure(parallel_read, [str(sys.argv[1]), int(sys.argv[2]) ])
