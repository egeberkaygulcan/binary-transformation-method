import numpy as np
from tqdm import tqdm
from river import optim
from river import utils
from river import stream
from river import metrics
from src import BinaryTransformation
from skmultilearn.dataset import load_from_arff

if __name__ == '__main__':
    # Load dataset from arff
    dataset = '20NG.arff'
    n_labels = 20

    X, Y = load_from_arff(f'./datasets/{dataset}', label_count=n_labels)
    X = X.toarray()
    Y = Y.toarray()

    # Define model
    model = BinaryTransformation(n_labels=n_labels)

    # Initialize model and metrics
    ds = stream.iter_array(X, Y)
    accuracy = metrics.ExactMatch()

    # Start prequential test-then-train run
    pbar = tqdm(total=len(X))
    for x, y in ds:
        # Test for the current sample
        y_pred = model.predict_one(x)
        y_pred = utils.numpy2dict(y_pred) # Output numpy array is transformed to dict for river compatibility

        # Train with the current sample
        model.learn_one(x, y)

        # Update metric
        accuracy.update(y, y_pred)

        pbar.update(1)
    pbar.close()

    print(accuracy)



