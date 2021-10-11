# welcome to the jungle! :)
# Anomalous dice detection script.
# 3 different approaches with OpenCV and TensorFlow.
# made by Hakan, Lyes, Vadym.
# check out the repo here https://github.com/hakanErgin/faktion-usecase

# import sys  # to be cool

from utils.option_a_step2_only import classifier_a_step2 as op1

def main_flow(file: str = 'love ya!', method: int = 1, stats: bool = False) -> int:
    """
    Main program to judge the dice's fate:
    returns 0 if normal,
    returns 1 if anomalous.
    Possible method values are <1>, <2>, <3>.
    Set stats to True to see acuracy.
    """

    method = input("please, \
        type 1 for OpenCV absdiff method\n\
        type 2 for OpenCV contour method\n\
        type 3 for TensorFlow CNN method.")
    if method == 1:
        print('running method 1')
        # op1(file)


# if __name__ == "__main__":
#    main_flow()
