import dill as pickle

def save_figure(figure, fpath):
    # Pickle the data
    with open(fpath, 'wb') as pkl_file:
        pickle.dump(figure, pkl_file)

def load_figure(fpath):
    # Unpickle the data
    with open(fpath, 'rb') as pkl_file:
            fig = pickle.load(pkl_file)
    return fig