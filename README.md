# Лабораторная работа 4. Перцептрон 

Сигмоидальная (логистическая) функция активации **g(z)**:

<img src = "https://github.com/python-advance/sem6-lr4-Yalkinzsun/blob/master/img/sigmoid_function.png" height = "150" />

График данной функции:

<img src = "https://github.com/python-advance/sem6-lr4-Yalkinzsun/blob/master/img/sigmoid.png" height = "150" />

### Логическая операция *AND*

<img src = "https://github.com/python-advance/sem6-lr4-Yalkinzsun/blob/master/img/and.png" height = "150" />

Матрица Θ: [-30 20 20]

Результат h(x) будет положительным, только если x1 и x2 будут равны 1:

<img src = "https://github.com/python-advance/sem6-lr4-Yalkinzsun/blob/master/img/and2.png" height = "200" />

### Логическая операция *XNOR*

Необходимо выразить данную логическую операцию через AND, OR и NOT, добавить скрытый слой в перцептрон:

<img src = "https://github.com/python-advance/sem6-lr4-Yalkinzsun/blob/master/img/xnor_0.png" height = "150" />

<img src = "https://github.com/python-advance/sem6-lr4-Yalkinzsun/blob/master/img/xnor.png" height = "200" />


## Тестирование

Используются параметрические тесты pytest
Команда для запуска тестов в терминале: `python -m pytest -v test_main.py`

```Python
from main import perceptron_or_and, perceptron_not, perceptron_xor, perceptron_xnor
import pytest


@pytest.mark.parametrize("th0", list(range(-15, -9)))
@pytest.mark.parametrize("th1", list(range(18, 22)))
@pytest.mark.parametrize("th2", list(range(18, 22)))
def test_or(th0, th1, th2):
    assert perceptron_or_and([(0, 0), (0, 1), (1, 0), (1, 1)], [th0, th1, th2]) == [0, 1, 1, 1]

...

@pytest.mark.parametrize("description, theta_or, theta_and, theta_not", [
    ("forth", [-15, 18, 19], [-32, 18, 18], [4, -32]),
    ("fifth", [-10, 18, 18], [-32, 19, 20], [1, -30]),
    ("sixth", [-14, 18, 22], [-30, 21, 21], [4, -25])])
def test_xnor(description, theta_or, theta_and, theta_not):
    assert perceptron_xnor([(0, 0), (0, 1), (1, 0), (1, 1)], theta_or, theta_and, theta_not)[0] == [1, 0, 0, 1]
```

#### Результат тестирования:

<img src = "https://github.com/python-advance/sem6-lr4-Yalkinzsun/blob/master/img/tests.png" height = "300" />

#### Результат выполнения программы:

```
NOT
θ = [4, -34]
+---+------+
| x | h(x) |
+---+------+
| 0 |  1   |
| 1 |  0   |
+---+------+

OR
θ = [-9, 19, 20]
+----+----+------+
| x1 | x2 | h(x) |
+----+----+------+
| 0  | 0  |  0   |
| 0  | 1  |  1   |
| 1  | 0  |  1   |
| 1  | 1  |  1   |
+----+----+------+

AND
θ = [-29, 19, 19]
+----+----+------+
| x1 | x2 | h(x) |
+----+----+------+
| 0  | 0  |  0   |
| 0  | 1  |  0   |
| 1  | 0  |  0   |
| 1  | 1  |  1   |
+----+----+------+

XOR
+----+----+----+----+------+
| x1 | x2 | a1 | a2 | h(x) |
+----+----+----+----+------+
| 0  | 0  | 0  | 1  |  0   |
| 0  | 1  | 1  | 1  |  1   |
| 1  | 0  | 1  | 1  |  1   |
| 1  | 1  | 1  | 0  |  0   |
+----+----+----+----+------+

XNOR
+----+----+----+----+------+
| x1 | x2 | a1 | a2 | h(x) |
+----+----+----+----+------+
| 0  | 0  | 0  | 1  |  1   |
| 0  | 1  | 0  | 0  |  0   |
| 1  | 0  | 0  | 0  |  0   |
| 1  | 1  | 1  | 0  |  1   |
+----+----+----+----+------+
```
