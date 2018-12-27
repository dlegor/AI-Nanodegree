"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def Near(move, Zone):
    """ 
    Validate if a Movements (move) is in the board zone (Zone).

    Parameters 
    ----------
    move: Movements in board
    Zone: Zone o region in the board
    
    Return
    ------
    Boolean Value(True/ False)
    """
    for Move in Zone:
        if move in Move:
            return True    
    return False

def Percentage_of_Board_Filled(game):
    """
    Is Validated the percentage of board filled.
    
    Parameters
    ---------
    game: `isolation.Board`

    Return
    ------
    int
       The int returned is the percentage the number of quadrants empty. 

    """
    spaces_empty = game.get_blank_spaces()
    return round((len(spaces_empty) / 49) * 100)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    # This is the best heuristic 

    #Valitation 1
    if game.is_loser(player):
        return float("-inf")
    #Validation 2
    if game.is_winner(player):
        return float("inf")

    #Definition of areas in board
    # Detalles in heuritic_analysis.pdf

    Corner=[(0,0),(6,0),(0,6),(6,6)]
    Values_three=[(0,1),(1,0),(5,0),(6,1),(0,5),(1,6),(6,5),(5,6)]
    Values_four=[(0,2),(0,3),(0,4),(6,2),(6,3),(6,4),(2,0),(3,0),(4,0),(2,6),(3,6),(4,6),(1,1),(1,5),(5,1),(5,5)]
    Values_six=[(1,2),(1,3),(1,4),(5,2),(5,3),(5,4),(2,1),(3,1),(4,1),(2,5),(3,5),(4,5)]
    Center=[(2, i) for i in range(2,5)]+[(3, i) for i in range(2,5)]+[(4, i) for i in range(2,5)]
    
    #Legal movements for players
    agent_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    
    #Scores 1
    agent_score = 0
    opp_score = 0
    
    #Scores 2
    Local_agent_score = 0
    Local_opp_score = 0
    
    #Loops on legal movements
    for move in agent_moves:
        game_forecast=game.forecast_move(move)
        agent_score=agent_score+len(game_forecast.get_legal_moves(player))

    for move in opponent_moves:
        game_forecast=game.forecast_move(move)
        opp_score=opp_score+len(game_forecast.get_legal_moves(game_forecast.get_opponent(player)))    

    #Loop on Legal Movements
    for move in agent_moves:
        if Near(move,Center):
            Local_agent_score+=10
        elif Near(move,Values_six):
            Local_agent_score+=7
        elif Near(move,Values_four):         
            Local_agent_score+=5
        elif Near(move,Values_three):
            Local_agent_score-=5
        elif Near(move,Corner):
            Local_agent_score-=10            
            
    for move in opponent_moves:
        if Near(move,Center):
            Local_opp_score+=10
        elif Near(move,Values_six):
            Local_opp_score+=7
        elif Near(move,Values_four):
            Local_opp_score+=5
        elif Near(move,Values_three):
            Local_opp_score-=5
        elif Near(move,Corner):
            Local_opp_score-=10
    
    #Penalty by localization  
    Local_Position_agent=game.get_player_location(player)
    Local_Position_opp=game.get_player_location(game.get_opponent(player))

    Local_Position_agent_score=0
    Local_Position_opp_score=0

    if Near(Local_Position_agent,Center):
        Local_Position_agent_score=10
    elif Near(Local_Position_agent,Values_six):
        Local_Position_agent_score=7
    elif Near(Local_Position_agent,Values_four):
        Local_Position_agent_score=5
    elif Near(Local_Position_agent,Values_three):
        Local_Position_agent_score=-5
    elif Near(Local_Position_agent,Corner):
        Local_Position_agent_score=-10

    if Near(Local_Position_opp,Center):
        Local_Position_opp_score=10
    elif Near(Local_Position_opp,Values_six):
        Local_Position_opp_score=7
    elif Near(Local_Position_opp,Values_four):
        Local_Position_opp_score=5
    elif Near(Local_Position_opp,Values_three):
        Local_Position_opp_score=-5
    elif Near(Local_Position_opp,Corner):
        Local_Position_opp_score=-10
    
    #Combiner lineal of scores
    return float((agent_score - opp_score)+(Local_agent_score- Local_opp_score)+(Local_Position_agent_score- Local_Position_opp_score))    
    
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!

    #Validation 1
    if game.is_loser(player):
        return float("-inf")
    #Validation 2
    if game.is_winner(player):
        return float("inf")

    #Legal Movements
    agent_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    #Scores
    agent_score = 0
    opp_score = 0

    for move in agent_moves:
        game_forecast=game.forecast_move(move)
        agent_score=agent_score+len(game_forecast.get_legal_moves(player))

    for move in opponent_moves:
        game_forecast=game.forecast_move(move)
        opp_score=opp_score+len(game_forecast.get_legal_moves(game_forecast.get_opponent(player)))    
    
    #Return the difference in scores
    return float(agent_score - opp_score)

    

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    #Validation 1
    if game.is_loser(player):
        return float("-inf")
    
    #Validation 2
    if game.is_winner(player):
        return float("inf")
    
    #Manhattan Distance
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)

    return float(abs(h - y) + abs(w - x))
    
    #################
    #Other Huristic #
    #################
    
    #def custom_score_3(game, player):
    
    #Validation 1
    #if game.is_loser(player):
    #    return float("-inf")
    
    #Validation 2
    #if game.is_winner(player):
    #    return float("inf")
     
    #Border = [(0, i) for i in range(game.width)]+[(i, 0) for i in range(game.height)]+[(game.width - 1, i) for i in range(game.width)]+[(i, game.height - 1) for i in range(game.height)]    
    
    #Legal Movements
    #agent_moves = game.get_legal_moves(player)
    #opp_moves = game.get_legal_moves(game.get_opponent(player))
    #
    # for move in agent_moves:
    #     if Percentage_of_Board_Filled(game) <= 10:
    #         agent_score += 1
    #     elif Percentage_of_Board_Filled(game) > 10 and Percentage_of_Board_Filled(game) <= 30 and Near_to_Limits(move, Border):
    #         agent_score += 5
    #     elif Percentage_of_Board_Filled(game) > 30 and Percentage_of_Board_Filled(game) <= 50 and Near_to_Limits(move, Border):
    #         agent_score += 10    
    #     elif Percentage_of_Board_Filled(game) > 50 and Percentage_of_Board_Filled(game) <= 70 and Near_to_Limits(move, Border):
    #         agent_score += 15
    #     elif Percentage_of_Board_Filled(game) > 70 and Percentage_of_Board_Filled(game) <= 90 and Near_to_Limits(move, Border):
    #         agent_score += 10    
    #     elif Percentage_of_Board_Filled(game) > 10 and Near_to_Limits(move, Border):
    #         agent_score += 5
    #     elif not Near_to_Limits(move, Border):
    #         agent_score += 1

    # for move in opponent_moves:
    #     if Percentage_of_Board_Filled(game) < 10:
    #         opp_score += 1
    #     elif Percentage_of_Board_Filled(game) > 10 and Percentage_of_Board_Filled(game) <= 30 and Near_to_Limits(move, Border):
    #         opp_score += 5
    #     elif Percentage_of_Board_Filled(game) > 30 and Percentage_of_Board_Filled(game) <= 50 and Near_to_Limits(move, Border):
    #         opp_score += 10    
    #     elif Percentage_of_Board_Filled(game) > 50 and Percentage_of_Board_Filled(game) <= 70 and Near_to_Limits(move, Border):
    #         opp_score += 15    
    #     elif Percentage_of_Board_Filled(game) > 70 and Percentage_of_Board_Filled(game) <= 90 and Near_to_Limits(move, Border):
    #         opp_score += 10        
    #     elif Percentage_of_Board_Filled(game) > 80 and Near_to_Limits(move, Border):
    #         opp_score += 5
    #     elif not Near_to_Limits(move, Border):
    #         opp_score += 1
    #Return the difference in scores
    #  return float(agent_score - opp_score)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # The Functions are a modification of the version of MINIMAX-DECISION of the AIMA text
        # The idea is similar.

        def max_value(game, depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0:
                return self.score(game, self)

            V = float('-inf')
            
            for move in game.get_legal_moves():             
                score = min_value(game.forecast_move(move), depth - 1)
                V = max(score,V)
            return V

        def min_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0:
                return self.score(game, self)

            V = float('inf')

            for move in game.get_legal_moves():
                score = max_value(game.forecast_move(move), depth - 1)
                V = min(score,V)

            return V

        #Code Main 
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if len(game.get_legal_moves())==0:
                return (-1,-1)
    

        main_score = float('-inf')
        best_move = (-1,-1)
        
        
        for move in game.get_legal_moves():            
            score = min_value(game.forecast_move(move), depth-1)
            if score >= main_score:
                best_move = move
                main_score = score

        return best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        
        #Initialization of depth
        depth = 1

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while True:
                best_move = self.alphabeta(game,depth)
                #if best_move== (-1,-1):
                #    break
                depth += 1
            else:
                best_move=self.alphabeta(game, self.search_depth)    
           
        except SearchTimeout:
            return best_move

            pass  # Handle any actions required after timeout as needed
            
        # Return the best move from the last completed search iteration
        return best_move
 

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        
        # Similar division in function as in MiniMax Function

        def max_value(game, depth,alpha,beta):
            #Terminal- Test
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 :
                return self.score(game, self)
            
            V = float('-inf')

            for move in game.get_legal_moves():             
                score = min_value(game.forecast_move(move), depth - 1,alpha,beta)
                if score==None:
                    score=float('-inf')
                V = max(score,V)
                
                if V>=beta:
                    return V
                alpha=max(alpha,V)
                
            return V    
            

        def min_value(game, depth,alpha,beta):
            #Terminal-Test
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0:
                return self.score(game, self)

            V = float('inf')


            for move in game.get_legal_moves():
                score = max_value(game.forecast_move(move), depth - 1,alpha,beta)
                if score==None:
                    score=float('inf')

                V = min(score,V)
                if V <= alpha:
                    return V
                beta=min(beta,V)    

            return V

        #Code Main 

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        
        main_score = float('-inf')
        best_move = (-1,-1)
        
        for move in game.get_legal_moves():            
            score = min_value(game.forecast_move(move), depth-1,alpha,beta)
            if score==None:
               score=float('-inf')

            if score > main_score:
                best_move = move
                main_score = score
           
            if main_score>=beta:
               return best_move
            
            alpha=max(alpha,main_score)    
        
        return best_move

