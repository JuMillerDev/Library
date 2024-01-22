import pickle

def save_model( model,file_name: str = 'model'):
    with open('{}.pkl'.format(file_name), 'wb') as file:
        pickle.dump(model,file)

def load_model(file_name: str = 'model'):
    with open('{}.pkl'.format(file_name),'rb') as file:
        return pickle.load(file)