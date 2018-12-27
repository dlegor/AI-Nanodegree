import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
 #  ref Udacity:
 #
 #  for each word_id:
 #   for each word model:
 #          calculate the scores for each word for each model and update the 'probabilities' list.
 #   determine the maximum score for each model.
 #   Append the corresponding word (the tested word is deemed to be the word for which with the model was trained) to the list 'guesses'.
 #
 #  https://discussions.udacity.com/t/recognizer-implementation/234793/3

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    probabilities = []
    guesses = []
    
    
    for test in range(0,test_set.num_items):
      X,lengths = test_set.get_all_Xlengths()[test]
      probability_dict = {}
      #max_score,max_word = float('-inf'),float('-inf')
      max_score,max_word =float('-inf'), float('-inf')
      for word,model in models.items():
          try:
              score = model.score(X,lengths)
          except:
              score = float('-inf')

          if max_score == float('-inf') or score > max_score:
                max_score = score
                max_word = word
          probability_dict[word] = score    

      probabilities.append(probability_dict)
      guesses.append(max_word)

    return (probabilities , guesses)
    