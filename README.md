# NN_2D_heat_equation_python
Machine learning model implementation to solve heat equation on a 2D rectangular surface

-create data to train network in data_create.py -> generates data points of temperature ranges in cell center and sum of surrounding 4 cells, output is the new temperature inside the cell

-train network in neural_net_train.py -> trains the neural network using inputs generated and stored in input.csv

-test model behaviour in data_predict.py -> with given constant wall BCs and random initial temperatures calculate the behaviour of HE:
![image](https://user-images.githubusercontent.com/94861828/171612538-74f750fc-dc0b-4782-9be2-ccc582a9a474.png)
