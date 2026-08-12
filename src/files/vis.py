# Copyright Author: Dr Tang Tiong Yew
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def vis2d(ax, model, X_train, Y_train, X_test=[], Y_test=[]):
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib is required for vis2d visualization.")
        return

    # Identify graph range
    x_range = [X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5]
    y_range = [X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5]
    if len(X_test) > 0:
        x_range = [min(x_range[0], X_test[:, 0].min() - 0.5), max(x_range[1], X_test[:, 0].max() + 0.5)]
        y_range = [min(y_range[0], X_test[:, 1].min() - 0.5), max(y_range[1], X_test[:, 1].max() + 0.5)]

    # Create meshgrid
    xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1], .01), np.arange(y_range[0], y_range[1], .01))
    
    # Identify area of decision
    Z = model.predict([[x, y] for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Plot decision areas
    ax.contourf(xx, yy, Z, alpha=.8)
    
    # Plot training and testing data
    ax.scatter([x[0] for x in X_train], [x[1] for x in X_train], c=Y_train, edgecolors='black')
    if len(X_test) > 0:
        ax.scatter([x[0] for x in X_test], [x[1] for x in X_test], c=Y_test, edgecolors='brown', alpha=.8)


def vis3d(fig, model, X_train, Y_train, X_test=[], Y_test=[]):
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib is required for vis3d visualization.")
        return []

    possible_class = np.unique(Y_train)
    y_range = [0, 1]
    y_data_min = X_train.min(axis=0)
    y_data_max = X_train.max(axis=0)
    if len(X_test) > 0:
        y_data_min = np.amin([y_data_min, X_test.min(axis=0)], axis=0)
        y_data_max = np.amax([y_data_max, X_test.max(axis=0)], axis=0)

    single_y = np.arange(y_range[0], y_range[1], .1)
    single_y = single_y.reshape(len(single_y), 1)
    yy = []
    for i in range(X_train.shape[1]):
        if len(yy) == 0:
            yy = np.tile(single_y, 1)
        else:
            old = np.tile(yy, (single_y.shape[0], 1))
            new = np.repeat(single_y, yy.shape[0])
            new = new.reshape(len(new), 1)
            yy = np.hstack([new, old])

    yy_data = [[yi * (y_data_max[i] - y_data_min[i]) + y_data_min[i] for i, yi in enumerate(y)] for y in yy]
    zz = model.predict(yy_data)
    train_x = (X_train - y_data_min) / (y_data_max - y_data_min + 1e-9)

    axes = []
    for i in possible_class:
        ax = fig.add_subplot(len(possible_class), 1, i + 1)
        ax.plot(yy[zz == i].transpose(), c=cm.Set2.colors[i % cm.Set2.N], alpha=0.5)
        ax.plot(train_x[Y_train == i].transpose(), c='black', lw=5, alpha=.8)
        ax.plot(train_x[Y_train == i].transpose(), c=cm.Dark2.colors[i % cm.Dark2.N], lw=3, alpha=.8)
        ax.set_title(f"output = {i}")
        ax.set_xticks([i for i in range(X_train.shape[1])])
        ax.set_ylim(y_range)
        axes.append(ax)

    return axes


if __name__ == '__main__':
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib is required to run the vis.py demonstration.")
        print("    Install it via: pip install matplotlib scikit-learn")
    else:
        try:
            from sklearn import datasets
            from sklearn.neural_network import MLPClassifier

            iris = datasets.load_iris()
            X = iris.data[:, :2]
            Y = iris.target
            mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000, random_state=42)
            mlp.fit(X, Y)

            fig, ax = plt.subplots(figsize=(6, 5))
            vis2d(ax, mlp, X, Y)
            ax.set_title("vis.py 2D Decision Boundary Self-Test")
            plt.tight_layout()
            print("[SUCCESS] vis.py module self-test executed cleanly.")
            plt.show()
        except Exception as e:
            print(f"[!] Error during vis.py self-test: {e}")