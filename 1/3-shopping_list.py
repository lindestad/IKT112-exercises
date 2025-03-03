def print_shopping_list(shopping_list: list):
    print("\n##### SHOPPING LIST #####")
    for idx, item in enumerate(shopping_list):
        print(f"{idx + 1}: {item}")


def main():
    print(
        "Welcome to my shopping list app! To add an item, simply type it. To print the shopping list and exit, type 'print'."
    )
    shopping_list = []
    usr_input = ""
    while usr_input != "print":
        usr_input = input()
        if usr_input != "print" and usr_input != "":
            shopping_list.append(usr_input)
    print_shopping_list(shopping_list)


if __name__ == "__main__":
    main()
