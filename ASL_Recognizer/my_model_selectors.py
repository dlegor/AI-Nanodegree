import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences



class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    LogL= likelihood of the fitted model
    p= numbers parameters
    logN= Number data points

    """

    def select(self):
        """ Select best model for self.this_word based on BIC score
        for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # List with BIC
        Criterion_BICs = []

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states) #Estimation hmm model
                LogL = hmm_model.score(self.X, self.lengths) # Likelihood 
                p=   (num_states**2)+(2*num_states * len(self.lengths))-1 # Numbers parameters  
                BIC = (-2*LogL)+(p*np.log(len(self.X))) # BIC
                Criterion_BICs.append((BIC, hmm_model))
            except:
                pass
            Criterion_BICs=list(filter(lambda x: x[0]!=None, Criterion_BICs)) #Filter on Scores
                
        return min(Criterion_BICs,key = lambda x: x[0])[1] if Criterion_BICs else None         

        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
 #   def log_likelihood_All_Rest_words(self, model, other_words):
 #       return [model.score(X, y) for X,y in other_words]

    def select(self):
        """ Select best model for self.this_word based on DIC score
        for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Selects rest of the words
        All_Rest_Words = [self.hwords[word] for word in self.words if word != self.this_word]
        # List DIC
        Criterion_DICs=[]

        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = self.base_model(num_states) # Fit Model
                logL = hmm_model.score(self.X, self.lengths)#Likelihood
                #Likelihood the Rest of the Words
                All_logL = np.sum([hmm_model.score(X, y) for X,y in All_Rest_Words])                  
                DIC =  logL - (All_logL / (len(self.words) - 1)) # Calculate DIC Score
                Criterion_DICs.append((DIC,hmm_model)) #Add List
            except:
                    pass
        Criterion_DICs=list(filter(lambda x: x[0]!=None, Criterion_DICs)) #Filter on Scores        

        return max(Criterion_DICs,key=lambda x: x[0])[1] if Criterion_DICs else None




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        """ Select best model for self.this_word based on Cross-Validation score
        for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object

        Ref:
        Gilles Celeux, Jean-Baptiste Durand/Selecting Hidden Markov Model State Number with Cross-Validated Likelihood
        http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
        Friedman-Hastie-Tibshirani/The Elements of Statistical Learning/Cap 7
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
            
        
        K_Fold = KFold(n_splits = 3, shuffle = False, random_state = None) #KFolds with 3 splits
        
        logL = []
        Criterion_CV = []

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                if len(self.sequences) > 2: 
                    for train_index, test_index in K_Fold.split(self.sequences):
                        
                        self.X, self.lengths = combine_sequences(train_index, self.sequences)
                        X_test, lengths_test = combine_sequences(test_index, self.sequences)
                        
                        hmm_model = self.base_model(num_states)
                        log_likelihood = hmm_model.score(X_test, lengths_test) #Score
                else:
                    hmm_model = self.base_model(num_states)
                    log_likelihood = hmm_model.score(self.X, self.lengths)#Score

                logL.append(log_likelihood)
                Criterion_CV.append((np.mean(logL),hmm_model)) #Average Log Likelihood of CV fold

            except Exception as e:
                pass

            Criterion_CV=list(filter(lambda x: x[0]!=None, Criterion_CV)) #Filter on Scores        
        return max(Criterion_CV, key = lambda x: x[0])[1] if Criterion_CV else None
