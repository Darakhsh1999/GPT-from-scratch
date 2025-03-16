import os.path as osp

FILE_PATH = osp.join("..","data","txt","1984.txt")

def load_text():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def get_characters_from_text(text, verbose=False):
    vocab_chars: list = sorted(list(set(text)))
    vocab_size = len(vocab_chars)
    if verbose:
        print("".join(vocab_chars))
        print(f"Vocab size of {vocab_size} characters")
    return vocab_chars
    



if __name__ == "__main__":

    text = load_text()
    print(len(text))
    print(text[:200])

    vocab = get_characters_from_text(text, verbose=True)