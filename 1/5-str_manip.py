def main():
    print("Enter a string!")
    usr_str = input()
    print(f"Uppercase: {usr_str.upper()}")
    print(f"Lowercase: {usr_str.lower()}")
    usr_str_rev = usr_str[::-1]

    print(f"The string is{" not"*(usr_str!=usr_str_rev)} a palindrome!")


if __name__ == "__main__":
    main()
