def convert(temp: float, target: int) -> float:
    match target:
        case 1:  # To Fahrenheit
            return (temp * (9 / 5)) + 32
        case 2:  # To Celcius
            return (temp - 32) * (5 / 9)


class FailedUserInput(Exception):
    pass


def get_user_input() -> tuple[float, int]:
    print("Enter a temperature:")
    temp = 0
    try:
        usr_input = input()
        temp = float(usr_input)
    except ValueError:
        print(f"That was not a valid number (unable to cast '{usr_input}' to float).")
        raise FailedUserInput()
    print("Select temperature type to convert to:\n1: Fahrenheit\n2: Celcius")
    target = 0
    usr_input = input()
    match usr_input:
        case "1" | "fahrenheit" | "Fahrenheit":
            target = 1
        case "2" | "celcius" | "Celcius":
            target = 2
        case _:
            print(f"Invalid entry, unable to convert to '{usr_input}'.")
            raise FailedUserInput()
    return (temp, target)


def main():
    print(
        "Welcome to my temperature converter!\nYou will be asked to enter a temperature value like '32.7', then to select wether to convert it to Fahrenheit or Celcius."
    )
    unsuccessful = True
    while unsuccessful:
        try:
            (temp, target) = get_user_input()
            result = convert(temp, target)
            match target:
                case 1:
                    print(f"\n{temp}°C converted to Fahreheit is: {result}.")
                case 2:
                    print(f"\n{temp}°F converted to Celcius is: {result}.")
            unsuccessful = False
        except FailedUserInput:
            print("\nLet's try again.")


if __name__ == "__main__":
    main()
