def check_password(password: str) -> bool:
    if len(password) < 9:
        return False

    upper, lower, number = False, False, False
    for c in password:
        if c.isupper():
            upper = True
        if c.islower():
            lower = True
        if c.isnumeric():
            number = True
    return upper and lower and number


def main():
    print("Enter a password:")
    usr_password = input()
    is_good = check_password(usr_password)
    print(f"That is a {("good" * is_good) + ("bad" * (not is_good))} password.")


if __name__ == "__main__":
    main()
