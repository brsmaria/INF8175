import nn
from backend import (DigitClassificationDataset, PerceptronDataset,
                     RegressionDataset)


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if (nn.as_scalar(self.run(x)) >= 0): return 1
        return -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        needToTrain = True
        while (needToTrain):
            needToTrain = False
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                if (y.data != prediction):
                    self.w.update(x, nn.as_scalar(y))
                    needToTrain = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    -> Pour approximer la fonction sinus, on a une caractéristique d'entrée (x)
    et un label de sortie (sin(x))
    """
    DIMENSION_COUCHE_CACHEE_1 = 10
    DIMENSION_COUCHE_CACHEE_2 = 20
    BATCH_SIZE = 100
    LOSS_THRESHOLD = 0.01
    LEARNING_RATE = 0.001

    def __init__(self) -> None:
        #Couche cachée 1
        self.w1 = nn.Parameter(1, self.DIMENSION_COUCHE_CACHEE_1)
        self.b1 = nn.Parameter(1, self.DIMENSION_COUCHE_CACHEE_1)

        #Couche cachée 2
        self.w2 = nn.Parameter(self.DIMENSION_COUCHE_CACHEE_1, self.DIMENSION_COUCHE_CACHEE_2)
        self.b2 = nn.Parameter(1, self.DIMENSION_COUCHE_CACHEE_2)
        
        #Couche de sortie
        self.w3 = nn.Parameter(self.DIMENSION_COUCHE_CACHEE_2, 1)
        self.b3 = nn.Parameter(1, 1)


    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        #Couche cachée 1
        l1 = nn.Linear(x, self.w1)
        p1 = nn.AddBias(l1, self.b1)
        n1 = nn.ReLU(p1)

        #couche cachée 2
        l2 = nn.Linear(n1, self.w2)
        p2 = nn.AddBias(l2, self.b2)
        n2 = nn.ReLU(p2)

        #Couche de sortie
        l3 = nn.Linear(n2, self.w3)
        p3 = nn.AddBias(l3, self.b3)
        return p3
        

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        loss_value = float('inf')
        while True:
            for x, y in dataset.iterate_once(self.BATCH_SIZE):
                loss_node = self.get_loss(x, y)
                loss_value = nn.as_scalar(loss_node)

                if loss_value <= self.LOSS_THRESHOLD:
                    return

                gradients = nn.gradients(loss_node, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])

                #couche cachée 1
                self.w1.update(gradients[0], -self.LEARNING_RATE)
                self.b1.update(gradients[1], -self.LEARNING_RATE)

                #couche cachée 2
                self.w2.update(gradients[2], -self.LEARNING_RATE)
                self.b2.update(gradients[3], -self.LEARNING_RATE)

                #couche de sortie
                self.w3.update(gradients[4], -self.LEARNING_RATE)
                self.b3.update(gradients[5], -self.LEARNING_RATE)



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    DIMENSION_COUCHE_CACHEE_1 = 128
    DIMENSION_COUCHE_CACHEE_2 = 64
    DIMENSION_COUCHE_SORTIE = 10
    BATCH_SIZE = 100
    LOSS_THRESHOLD = 0.01
    LEARNING_RATE = 0.1

    def __init__(self) -> None:
        # Couche cachée 1
        self.w1 = nn.Parameter(784, self.DIMENSION_COUCHE_CACHEE_1)
        self.b1 = nn.Parameter(1, self.DIMENSION_COUCHE_CACHEE_1)

        # Couche cachée 2
        self.w2 = nn.Parameter(self.DIMENSION_COUCHE_CACHEE_1, self.DIMENSION_COUCHE_CACHEE_2)
        self.b2 = nn.Parameter(1, self.DIMENSION_COUCHE_CACHEE_2)
        
        # Couche de sortie
        self.w3 = nn.Parameter(self.DIMENSION_COUCHE_CACHEE_2, self.DIMENSION_COUCHE_SORTIE)
        self.b3 = nn.Parameter(1, self.DIMENSION_COUCHE_SORTIE)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # Couche cachée 1
        l1 = nn.Linear(x, self.w1)
        p1 = nn.AddBias(l1, self.b1)
        n1 = nn.ReLU(p1)

        # Couche cachée 2
        l2 = nn.Linear(n1, self.w2)
        p2 = nn.AddBias(l2, self.b2)
        n2 = nn.ReLU(p2)

        # Couche de sortie
        l3 = nn.Linear(n2, self.w3)
        p3 = nn.AddBias(l3, self.b3)
        return p3

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        prediction = self.run(x)
        return nn.SoftmaxLoss(prediction, y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(self.BATCH_SIZE):
                loss_node = self.get_loss(x, y)

                gradients = nn.gradients(loss_node, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])

                # Couche cachée 1
                self.w1.update(gradients[0], -self.LEARNING_RATE)
                self.b1.update(gradients[1], -self.LEARNING_RATE)

                # Couche cachée 2
                self.w2.update(gradients[2], -self.LEARNING_RATE)
                self.b2.update(gradients[3], -self.LEARNING_RATE)

                # Couche de sortie
                self.w3.update(gradients[4], -self.LEARNING_RATE)
                self.b3.update(gradients[5], -self.LEARNING_RATE)

            if dataset.get_validation_accuracy() >= 0.97:
                break