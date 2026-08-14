# Copyright Author: Dr Tang Tiong Yew
def tossFairCoin():
  options = ['head', 'tail']
  headOrTail = ""
  if random.random() < 0.5:
    headOrTail = options[0]
  else:
    headOrTail = options[1]
  return headOrTail

def tossUnfairCoin():
  options = ['head', 'tail']
  headOrTail = ""
  if random.random() < 0.2:
    headOrTail = options[0]
  else:
    headOrTail = options[1]
  return headOrTail

def chooseFromThree():
  options = ['A', 'B', 'C']
  probabilities = [0.2, 0.5, 0.3]
  selectedOptions = ''
  selectedProbability = random.random()
  currentCumulatedProbability = 0
  for opt, prob in zip(options, probabilities):
    currentCumulatedProbability += prob
    if selectedProbability < currentCumulatedProbability:
      selectedOptions = opt
      break
  return selectedOptions

if __name__ == "__main__":
  print("Tossing a fair coin: {}".format(tossFairCoin()))
  print("Tossing an unfair coin: {}".format(tossUnfairCoin()))
  print("Choosing from an option of three: {}".format(chooseFromThree()))