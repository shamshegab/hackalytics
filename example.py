from load_model import load_model
model = load_model()
def main():
    result = model("My name is Michael. I go to school by bus.")
    print(str(result))
    return str(result)
main()
