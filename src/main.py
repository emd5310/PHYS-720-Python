import sys
import homework_one as hw1
import homework_three as hw3
import homework_four as hw4
import homework_five as hw5
import homework_six as hw6
import homework_seven as hw7
# The main program calling all homework problem functions


def main():
    homework_set = int(sys.argv[1])  # Takes the first argument in params as the homework set to execute
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("| PHYS-720 : Computational Methods for Physics |")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if homework_set == 1:
        hw1.run()
    elif homework_set == 3:
        hw3.run()
    elif homework_set == 4:
        hw4.run()
    elif homework_set == 5:
        hw5.run()
    elif homework_set == 6:
        hw6.run()
    elif homework_set == 7:
        hw7.run()
    else:
        print("Invalid set!")


if __name__ == '__main__':
    main()
