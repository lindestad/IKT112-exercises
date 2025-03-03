# This program _intentionally_ calculates wether or not
# a given year is a leap year to follow the assignment
# text. Dates are complex and this should be solved
# with an external library:
# import calendar
# calendar.isleap(my_year)


def is_leap_year(year) -> bool:
    return not (year % 400 or (year % 4 and not year % 100))


def main():
    print("Enter a year:")
    year = input()
    try:
        year = int(year)
    except ValueError:
        print(f"Invalid entry: '{year}'. Can't cast to int.")
        return

    print(f"The year {year} is{" not"*(not is_leap_year(year))} a leap year.")


if __name__ == "__main__":
    main()
