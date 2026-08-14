# Copyright Author: Dr Tang Tiong Yew
import numpy as np
import matplotlib.pyplot as plt

def bin2gray(b):
  for i,x in enumerate(b):
    if i == 0:
      g = x
    else:
      g = g + "{:d}".format((b[i-1] != b[i]))
  return g

def hamming(x1, x2):
  h_dist = 0
  for xi1, xi2 in zip(x1, x2):
    h_dist += (xi1 != xi2)
  return h_dist

if __name__ == "__main__":
  # for my_dec in range(16):
  #   my_bin = np.binary_repr(my_dec,4)
  #   my_gray = bin2gray(my_bin)
  #   print(my_dec, my_bin, my_gray)
  
  my_dec = list(range(16))
  my_bin = [np.binary_repr(x,4) for x in my_dec]
  my_gray = [bin2gray(x) for x in my_bin]
  my_bin_h_dist = [hamming(x1,x2) for x1,x2 in zip(my_bin[:-1], my_bin[1:])]
  my_gray_h_dist = [hamming(x1,x2) for x1,x2 in zip(my_gray[:-1], my_gray[1:])]

  plt.plot(my_bin_h_dist, label="Binary code")
  plt.plot(my_gray_h_dist, label="Gray code")
  plt.title("Hamming distances")
  plt.xlabel("Decimal value")
  plt.ylabel("Hamming distance")
  plt.legend()

  plt.show()
