# whole_history_rating
a python convertion from the ruby implementation of Rémi Coulom's Whole-History Rating (WHR) algorithm.

the original code can be found [here](https://github.com/goshrine/whole_history_rating)


Usage
-----

    import base
    
    whr = base()
    
    # base.create_game() arguments: black player name, white player name, winner, day number, handicap
    # Handicap should generally be less than 500 elo
    whr.create_game("shusaku", "shusai", "B", 1, 0)
    whr.create_game("shusaku", "shusai", "W", 2, 0)
    whr.create_game("shusaku", "shusai", "W", 3, 0)

    # Iterate the WHR algorithm towards convergence with more players/games, more iterations are needed.
    whr.iterate(50)
    
    # Or let the module iterate until the elo is stable (precision by default 10E-3) with a time limit of 10 seconds by default
    whr.auto_iterate(time_limit = 10, precision = 10E-3)

    # Results are stored in one triplet for each day: [day_number, elo_rating, uncertainty]
    whr.ratings_for_player("shusaku") => 
      [[1, -43, 84], 
       [2, -45, 84], 
       [3, -45, 84]]
    whr.ratings_for_player("shusai") => 
      [[1, 43, 84], 
       [2, 45, 84], 
       [3, 45, 84]]

    # You can print or get all ratings ordered
    whr.print_ordered_ratings()
    whr.get_ordered_ratings()

    # You can get a prediction for a future game between two players (even non existing players)
    # base.probability_future_match() arguments: black player name, white player name, handicap
    whr.probability_future_match("shusaku", "shusai",0) =>
      win probability: shusaku:37.24%; shusai:62.76%

