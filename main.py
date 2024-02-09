print("Please choose the neural network\n1) Convolutional Neural Network")
user_input = input("\nYour Answear: ")

match user_input:
    case "1": 
        pass
    case _: print("No Neural Network With Such Name")