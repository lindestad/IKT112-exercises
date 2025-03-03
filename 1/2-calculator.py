def calculator(a: float, operator: int, b: float) -> float:
    match operator:
        case 1:  # Addition
            return a + b
        case 2:  # Subtraction
            return a - b
        case 3:  # Multiplication
            return a * b
        case 4:  # Division
            if b != 0:
                return a / b
            else:
                raise ZeroDivisionError


class FailedUserInput(Exception):
    pass


def get_user_input() -> tuple[float, int, float]:
    print("Enter first number: ")
    a = input()
    try:
        a = float(a)
    except ValueError:
        print(f"That was not a valid number (unable to cast '{a}' to float).")
        raise FailedUserInput()

    print(
        "Enter operator:\n1: Addition (+)\n2: Subtraction (-)\n3: Multiplication (×)\n4: Division (÷)"
    )
    operator = input()
    valid_operators = ["1", "+", "2", "-", "3", "*", "×", "4", "/", "÷"]
    try:
        assert operator in valid_operators
        match operator:
            case "1" | "+":
                operator = 1
            case "2" | "-":
                operator = 2
            case "3" | "*" | "×":
                operator = 3
            case "4" | "/" | "÷":
                operator = 4
    except AssertionError:
        print(f"You entered an invalid operator: '{operator}'.")
        print(f"Valid choices are: ", end="")
        for valid_operator in valid_operators:
            print(f" '{valid_operator}'", end="")
        print(".\n", end="")
        raise FailedUserInput()

    print("Enter second number: ")
    b = input()
    try:
        b = float(b)
    except ValueError:
        print(f"That was not a valid number (unable to cast '{b}' to float).")
        raise FailedUserInput()
    return (a, operator, b)


def main():
    print(
        "Welcome to my calculator, you will be asked to enter a number, an operator, and a final number."
    )
    unsuccessful = True
    while unsuccessful:
        try:
            (a, operator, b) = get_user_input()
            answer = calculator(a, operator, b)
            print(f"Result of calculation: {answer}")
            unsuccessful = False
        except FailedUserInput:
            print("\nLet's try again.")
        except ZeroDivisionError:
            print("\nPlease don't try to divide by zero, let's try again.")


if __name__ == "__main__":
    main()
