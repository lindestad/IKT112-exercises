from numba import njit


class FailedUserInput(Exception):
    pass


def get_user_input() -> tuple[int, int]:
    choices = []
    for prompt_str in ["first", "second"]:
        print(f"Enter the {prompt_str} positive integer")
        i = input()
        try:
            i = int(i)
            assert i >= 0
            choices.append(i)
        except ValueError:
            print(f"Invalid input: '{i}'. Failed to cast to int.")
            raise FailedUserInput()
        except AssertionError:
            print(f"The integer must be positive!")
            raise FailedUserInput()
    return (choices[0], choices[1])


@njit  # Make it less painfully slow
def greatest_common_divisor(n: int, m: int) -> int:
    d = min(n, m)
    while n % d or m % d:
        d -= 1
    return d


def main():
    print("This program finds the greates common divisor among two positive integers.")
    unsuccessful = True
    while unsuccessful:
        try:
            (n, m) = get_user_input()
            unsuccessful = False
        except FailedUserInput:
            print("\nLet's try again!")

    solution = greatest_common_divisor(n, m)
    print(f"The greates common divisor of {n} and {m} is {solution}.")


if __name__ == "__main__":
    main()
