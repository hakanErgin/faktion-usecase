# Anomalous dice detection script.
# 3 different approaches with OpenCV and TensorFlow.
# made by Hakan, Lyes, Vadym.
# check out the repo here https://github.com/hakanErgin/faktion-usecase


# from utils.option_a_step2_only_copy import report as op1
from src.mixed_report import report as op3
from src.option_1 import classifier_a_step2 as op1
from src.classifier import detect_anomaly as op2


def main_flow(file: str = 'hi', method: int = 1, stats: bool = False) -> int:
    """
    Main program to judge the dice's fate:
    returns 0 if normal,
    returns 1 if anomalous.
    Possible method values are <1>, <2>, <3>.
    Set stats to True to see acuracy.
    """
    file = input("please, provide file name: ")
    method = int(input("please, \
type 1 for OpenCV absdiff method\n\
        type 2 for OpenCV contour method\n\
        type 3 for TensorFlow CNN method: "))
    if method == 1:
        print('running method 1')
        print(op1(file)[0])
    if method == 3:
        print('running method 3')
        op3()
    if method == 2:
        print('running method 2')
        op2(file)



if __name__ == "__main__":
    main_flow()
