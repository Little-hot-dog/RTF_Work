# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Сафаргалеев Никита Олегович
- РИ210914
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы "Helo World" на Python и Unity
- Для Python в отчете привести скриншоты с демонстрацией сохранения документа google.collab на свой диск с запуском программы, выводящей сообщение Hello World.
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/%D0%B3%D1%83%D0%B3%D0%BB%20%D0%B4%D0%B8%D1%81%D0%BA.png)
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/%D0%BA%D0%B0%D0%BB%D0%B0%D0%B1.png)
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/Hello_World_Py.png)
- Для Unity в отчете привести скриншоты вывода сообщения Hello World в консоль.
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/Hello_World_Unity.png)
## Задание 2
### Пошагово выполняю каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
- Произвожу подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py
import numpy as np
import matplotlib.pyplot as plt

x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)
plt.scatter(x,y)
```

- Определяю связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py

def model(a, b, x):
    return a*x + b
def loss_function(a, b, x, y):
    num = len (x)
    prediction = model(a,b,x)
    return (0.5/num) * (np.square(prediction-y)).sum()
def optimize(a,b,x,y):
    num = len(x)
    prediction = model(a, b, x)
    da = (1.0/num) * ((prediction - y)*x).sum()
    db = (1.0 / num) * ((prediction - y).sum())
    a = a - Lr*da
    b = b - Lr*db
    return a, b

def iterate(a,b,x,y,times):
    for i in range(times):
        f,b = optimize(a, b, x, y)
    return a,b

```

### Шаг_1 инициализация и модель итеративной оптимизации
```py
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

a,b = iterate(a,b,x,y,1)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
```py
[0.89586207]
[0.58350623]
[0.89852595] [0.58354363] 1313.5139238614338
```
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/%D1%88%D0%B0%D0%B3%201.png)

### Шаг_2 на второй итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации
```py
a,b = iterate(a,b,x,y,2)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
```py
[0.90382864] [0.58361804] 1299.473940949766
```
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/%D1%88%D0%B0%D0%B3%202.png)
### Шаг_3 третья итерация показывает значения параметров, значения потерь и визуализацию после итерации 
```py
a,b = iterate(a,b,x,y,3)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
```py
[0.91172039] [0.5837287] 1278.7424217414373
```
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/%D1%88%D0%B0%D0%B3%203.png)
### Шаг_4 на четвертой итерации отображаются значения параметров, значения потерь, и эффекты визуализации
```py
a,b = iterate(a,b,x,y,4)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
```py
[0.92212755] [0.58387449] 1251.7020616032517
```
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/%D1%88%D0%B0%D0%B3%204.png)
### Шаг_5 пятая итерация показывает значение параметра, значение потерь, и эффект визуализации после итерации
```py
a,b = iterate(a,b,x,y,5)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
```py
[0.93495371] [0.58405396] 1218.84456985552
```
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/%D1%88%D0%B0%D0%B3%205.png)
### Шаг_6 10000-я итерация показывает значения параметров, значения потерь и визуализацию после итерации 
```py
a,b = iterate(a,b,x,y,10000)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
```py
[1.74455547] [0.56478465] 190.49160221222198
```
![](https://github.com/Little-hot-dog/RTF_Work/blob/main/%D1%88%D0%B0%D0%B3%206.png)
## Задание 3
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.
loss может стремиться к нулю при условии, что ```b = 0``` и ```x = y``` или при очень маленьких значениях ```x, y```

```py
loss = loss_function(1,0,x,x)
```
```py
0.0
```

```
loss = loss_function(2,0,x*10**(-40), y*10**(-40))
```
```
2.843999999999999e-78
```
### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.
Чем больше ```Lr```, тем больше изменится значение ```a``` и ```b```. Чем меньше значение ```Lr```, тем меньше измменится ```a``` и ```b```.

```py


```

## Выводы

В ходе лабораторной работы я научился взаимодействовать с google.colab, просматривать графики и анализировать алгоритм работы программ. Кроме того научился работать в github и ознакомился с основными операторами зыка Python на примере реализации линейной регрессии.

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**

