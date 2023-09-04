import os
import shutil

def clear():
    delConfirm = input("clear workspace?[Y(yes)/N(no)]")
    if delConfirm == 'Y' or delConfirm == 'yes':
        print("remove source")
        shutil.rmtree("source")
        os.mkdir("source")
        print("remove map")
        shutil.rmtree("map")
        os.mkdir("map")
        print("remove finished")
        exit(0)
    elif delConfirm == 'N' or delConfirm == 'no':
        exit(0)
    else:
        print("you need input Y/yes or N/no")
        clear()

if __name__ == '__main__':
    clear()