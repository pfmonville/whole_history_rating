
# Whole History Rating (WHR) Python Implementation

This Python library is a conversion from the original Ruby implementation of Rémi Coulom's Whole-History Rating (WHR) algorithm, designed to provide a dynamic rating system for games or matches where players' skills are continuously estimated over time.

The original Ruby code is available here at [goshrine](https://github.com/goshrine/whole_history_rating).

## Installation

To install the library, use the following command:

```shell
pip install whole-history-rating
```

## Usage

### Basic Setup

Start by importing the library and initializing the base WHR object:

```python
from whr import whole_history_rating

whr = whole_history_rating.Base()
```

### Creating Games

Add games to the system using `create_game()` method. It takes the names of the black and white players, the winner ('B' for black, 'W' for white), the day number, and an optional handicap (generally less than 500 elo).

```python
whr.create_game("shusaku", "shusai", "B", 1, 0)
whr.create_game("shusaku", "shusai", "W", 2, 0)
whr.create_game("shusaku", "shusai", "W", 3, 0)
```


### Refining Ratings Towards Stability

To achieve accurate and stable ratings, the WHR algorithm allows for iterative refinement. This process can be controlled manually or handled automatically to adjust player ratings until they reach a stable state.

#### Manual Iteration

For manual control over the iteration process, specify the number of iterations you wish to perform. This approach gives you direct oversight over the refinement steps.

```python
whr.iterate(50)
```

This command will perform 50 iterations, incrementally adjusting player ratings towards stability with each step.

#### Automatic Iteration

For a more hands-off approach, the algorithm can automatically iterate until the Elo ratings stabilize within a specified precision. Automatic iteration is particularly useful when dealing with large datasets or when seeking to automate the rating process.

```python
whr.auto_iterate(time_limit=10, precision=1e-3, batch_size=10)
```

- `time_limit` (optional): Sets a maximum duration (in seconds) for the iteration process. If `None` (the default), the algorithm will run indefinitely until the specified precision is achieved.
- `precision` (optional): Defines the desired level of accuracy for the ratings' stability. The default value is `0.001`, indicating that iteration will stop when changes between iterations are less than or equal to this threshold.
- `batch_size` (optional): Determines the number of iterations to perform before checking for convergence and, if a `time_limit` is set, before evaluating whether the time limit has been reached. The default value is `10`, balancing between frequent convergence checks and computational efficiency.

This automated process allows the algorithm to efficiently converge to stable ratings, adjusting the number of iterations dynamically based on the complexity of the data and the specified precision and time constraints.


### Viewing Ratings

Retrieve and view player ratings, which include the day number, elo rating, and uncertainty:

```python
# Example output for whr.ratings_for_player("shusaku")
print(whr.ratings_for_player("shusaku"))
# Output:
#   [[1, -43, 0.84], 
#    [2, -45, 0.84], 
#    [3, -45, 0.84]]

# Example output for whr.ratings_for_player("shusai")
print(whr.ratings_for_player("shusai"))
# Output:
#   [[1, 43, 0.84], 
#    [2, 45, 0.84], 
#    [3, 45, 0.84]]

```

You can also view or retrieve all ratings in order:

```python
whr.print_ordered_ratings(current=False)  # Set `current=True` for the latest rankings only.
ratings = whr.get_ordered_ratings(current=False, compact=False)  # Set `compact=True` for a condensed list.
```

### Predicting Match Outcomes

Predict the outcome of future matches, including between non-existent players:

```python
# Example of predicting a future match outcome
probability = whr.probability_future_match("shusaku", "shusai", 0)
print(f"Win probability: shusaku: {probability[0]*100}%; shusai: {probability[1]*100}%")
# Output:
#   Win probability: shusaku: 37.24%; shusai: 62.76%  <== this is printed
#   (0.3724317501643667, 0.6275682498356332)
```


### Enhanced Batch Loading of Games

This feature facilitates the batch loading of multiple games simultaneously by accepting a list of strings, where each string encapsulates the details of a single game. To accommodate names with both first and last names and ensure flexibility in data formatting, you can specify a custom separator (e.g., a comma) to delineate the game attributes.

#### Standard Loading

Without specifying a separator, the default space (' ') is used to split the game details:

```python
whr.load_games([
    "shusaku shusai B 1 0",  # Game 1: Shusaku vs. Shusai, Black wins, Day 1, no handicap.
    "shusaku shusai W 2 0",  # Game 2: Shusaku vs. Shusai, White wins, Day 2, no handicap.
    "shusaku shusai W 3 0"   # Game 3: Shusaku vs. Shusai, White wins, Day 3, no handicap.
])
```

#### Custom Separator for Complex Names

When game details include names with spaces, such as first and last names, utilize the `separator` parameter to define an alternative delimiter, ensuring the integrity of each data point:

```python
whr.load_games([
    "John Doe, Jane Smith, W, 1, 0",  # Game 1: John Doe vs. Jane Smith, White wins, Day 1, no handicap.
    "Emily Chen, Liam Brown, B, 2, 0"  # Game 2: Emily Chen vs. Liam Brown, Black wins, Day 2, no handicap.
], separator=",")
```

This method allows for a clear and error-free way to load game data, especially when player names or game details include spaces, providing a robust solution for managing diverse datasets.


### Saving and Loading States

Save the current state to a file and reload it later to avoid recalculating:

```python
whr.save_base('path_to_save.whr')
whr2 = whole_history_rating.Base.load_base('path_to_save.whr')
```

## Optional Configuration

Adjust the `w2` parameter, which influences the variance of rating change over time, allowing for faster or slower progression. The default is set to 300, but Rémi Coulom used a value of 14 in his paper to achieve his results.

```python
whr = whole_history_rating.Base({'w2': 14})
```

Enable case-insensitive player names to treat "shusaku" and "ShUsAkU" as the same entity:

```python
whr = whole_history_rating.Base({'uncased': True})
```
