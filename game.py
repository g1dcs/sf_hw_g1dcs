""""Игра угадай число.
компьютер сам загадывает и угадывает
"""

from itertools import count
import numpy as np

def random_predict(number:int=1) -> int:
    """Рандомно угадываем число

    Args:
        numer (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Чмсло попыток
    """
    
    count =0
    
    while True:
        count += 1
        predict_number = np.random.randint(1, 101) # предполагаемое число
        if number == predict_number:
            break # выход из цикла, если угадалаи
    return(count)
