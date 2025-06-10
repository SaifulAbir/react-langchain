from dotenv import load_dotenv
load_dotenv()


def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    return len(text)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(get_text_length(text="Dog"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
