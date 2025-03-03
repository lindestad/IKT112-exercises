def main():
    # Source: Asked chatGPT to give me some valid english sample words
    sample_words = [
        "quiz",
        "jazz",
        "buzz",
        "fizz",
        "quick",
        "vex",
        "zephyr",
        "scrabble",
        "python",
        "example",
    ]

    point = dict()
    point_lists = [
        ["A", "E", "I", "L", "N", "O", "R", "S", "T", "U"],
        ["D", "G"],
        ["B", "C", "M", "P"],
        ["F", "H", "V", "W", "Y"],
        ["K"],
        ["J", "X"],
        ["Q", "Z"],
    ]
    for idx, letters in enumerate(point_lists):
        for letter in letters:
            point[letter] = idx + 1

    print("Calculating points of some example words:")
    for word in sample_words:
        point_counter = 0
        for letter in word:
            point_counter += point[letter.upper()]
        print(f"The word '{word}' is worth {point_counter} points.")


if __name__ == "__main__":
    main()
