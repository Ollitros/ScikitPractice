import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn import datasets
from Supervised import utils


y, x = utils.dowload_mnist_from_db()

print(y.shape, x.shape)