def main(path):
    uniqueChars = []
    with open(path) as file:
        for line in file.read():
            for char in line:
                if char not in uniqueChars:
                    uniqueChars.append(char)
    return uniqueChars


if __name__ == '__main__':
    uChars = main('../data/Gutenberg_eBook_de_oliver_twist.txt')
    print(uChars)
    print(len(uChars))  # 91, ohne \n 90, still one more then the JS unique values
