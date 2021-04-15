# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of the environments from 'World Discovery Models'[https://arxiv.org/pdf/1902.07685.pdf]. 
Learn to explore!

This environment uses a simple scrolling mechanism: cropping! As far as the pycolab engine is concerned, 
the game world doesn't scroll at all: it just renders observations that are the size
of the entire map. Only later do "cropper" objects crop out a part of the
observation to give the impression of a moving world/partial observability.

Keys: up, down, left, right - move. q - quit.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import curses

import sys
import numpy as np

from pycolab import ascii_art
from pycolab import cropping
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

# pylint: disable=line-too-long
MAZES_ART = [
    # Each maze in MAZES_ART must have exactly one of the object sprites
    # 'a', 'b', 'c', 'd' and 'e'. I guess if you really don't want them in your maze
    # can always put them down in an unreachable part of the map or something.
    #
    # Make sure that the Player will have no way to "escape" the maze.
    #
    # Legend:
    #     '#': impassable walls.            'a': moveable object A.
    #     'P': player starting location.    'b': white noise object B.
    #     ' ': boring old maze floor.
    #
    # Room layout:
    #   2
    # 1 0 3
    #   4

    # Maze #0: (paper: 5 rooms environment)
    ['###################',
     '##               ##',
     '# #             # #',
     '#  #     e     #  #',
     '#   #         #   #',
     '#    #### ####    #',
     '#    #### ####    #',
     '#    ##     ##    #',
     '#    ##     ##    #',
     '#        P        #',
     '#    ##     ##    #',
     '#    ##     ##    #',
     '#    #### ####    #',
     '#    #### ####    #',
     '#   #         #   #',
     '#  #           #  #',
     '# # b           # #',
     '##               ##',
     '###################']
]

# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # This is you, the player
             'e': (99, 140, 140),    # Patroller A
             'b': (145, 987, 341)}  # Patroller B

COLOUR_BG = {'@': (0, 0, 0)}  # So the coins look like @ and not solid blocks.

ENEMIES = {'e', 'b'} # Globally accessible set of sprites

# Empty coordinates corresponding to each numbered room (width 1 passageways not blocked)
ROOMS = {
  0 : [[7, 7], [7, 8], [7, 9], [7, 10], [7, 11], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11]],
  1 : [[4, 1], [4, 2], [5, 1], [5, 2], [5, 3], [6, 1], [6, 2], [6, 3], [6, 4], [7, 1], [7, 2], [7, 3], [7, 4], [8, 1], [8, 2], [8, 3], [9, 1], [9, 2], [9, 3], [10, 1], [10, 2], [10, 3], [11, 1], [11, 2], [11, 3], [11, 4], [12, 1], [12, 2], [12, 3], [12, 4], [13, 1], [13, 2], [13, 3], [14, 1], [14, 2]],
  2 : [[1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [4, 6], [4, 7], [4, 11], [4, 12]],
  3 : [[4, 16], [4, 17], [5, 15], [5, 16], [5, 17], [6, 14], [6, 15], [6, 16], [6, 17], [7, 14], [7, 15], [7, 16], [7, 17], [8, 15], [8, 16], [8, 17], [9, 15], [9, 16], [9, 17], [10, 15], [10, 16], [10, 17], [11, 14], [11, 15], [11, 16], [11, 17], [12, 14], [12, 15], [12, 16], [12, 17], [13, 15], [13, 16], [13, 17], [14, 16], [14, 17]],
  4 : [[14, 6], [14, 7], [14, 11], [14, 12], [15, 5], [15, 6], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12], [15, 13], [16, 4], [16, 5], [16, 6], [16, 7], [16, 8], [16, 9], [16, 10], [16, 11], [16, 12], [16, 13], [16, 14], [17, 4], [17, 5], [17, 6], [17, 7], [17, 8], [17, 9], [17, 10], [17, 11], [17, 12], [17, 13], [17, 14]],
}

def make_game(level):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  maze_ascii = MAZES_ART[level]

  return ascii_art.ascii_art_to_game(
      maze_ascii, what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
          'e': MoveableObject,
          'b': WhiteNoiseObject},
      update_schedule=['P', 'e', 'b'],
      z_order='ebP')

def make_croppers(level):
  """Builds and returns `ObservationCropper`s for the selected level.

  We make one cropper for each level: centred on the player. Room
  to add more if needed.

  Args:
    level: level to make `ObservationCropper`s for.

  Returns:
    a list of all the `ObservationCropper`s needed.
  """
  return [
      # The player view.
      cropping.ScrollingCropper(rows=5, cols=5, to_track=['P']),
  ]

class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player, the maze explorer."""

  def __init__(self, corner, position, character):
    """Constructor: just tells `MazeWalker` we can't walk through walls or objects."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#')
    self.last_position = None # store last position for moveable object
    self.last_action = None # store last action for moveable object

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, layers  # Unused

    self.last_position = self.position
    self.last_action = actions
    if actions == 0:    # go upward?
      self._north(board, the_plot)
    elif actions == 1:  # go downward?
      self._south(board, the_plot)
    elif actions == 2:  # go leftward?
      self._west(board, the_plot)
    elif actions == 3:  # go rightward?
      self._east(board, the_plot)
    elif actions == 4:  # stay put? (Not strictly necessary.)
      self._stay(board, the_plot)
    if actions == 5:    # just quit?
      the_plot.terminate_episode()

class WhiteNoiseObject(prefab_sprites.MazeWalker):
  """Randomly sample direction from left/right/up/down"""

  def __init__(self, corner, position, character):
    """Constructor: list impassables, initialise direction."""
    super(WhiteNoiseObject, self).__init__(corner, position, character, impassable='Pe#')
    # Initialize empty space in surrounding radius.
    self._empty_coords = ROOMS[4]

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.
    self._teleport(self._empty_coords[np.random.choice(len(self._empty_coords))])

class MoveableObject(prefab_sprites.MazeWalker):
  """Moveable object. Can be pushed by agent."""

  def __init__(self, corner, position, character):
    super(MoveableObject, self).__init__(corner, position, character, impassable='#b')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    mr, mc = self.position
    pr, pc = things['P'].last_position
    p_action = things['P'].last_action

    # move up
    if (mc == pc) and (mr - pr == -1) and (p_action == 0):
      moved = self._north(board, the_plot)
      if moved is not None:
        things['P']._south(board, the_plot)

    # move down
    elif (mc == pc) and (mr - pr == 1) and (p_action == 1):
      exiting_room = (self.position == (3, 9))
      if exiting_room == True:
        things['P']._north(board, the_plot)
        self._stay(board, the_plot)
      else:
        moved = self._south(board, the_plot)
        if moved is not None: # obstructed
          things['P']._north(board, the_plot)

    # move right
    elif (mc - pc == 1) and (mr == pr) and (p_action == 3):
      exiting_room = (self.position == (4, 8))
      if exiting_room == True:
        things['P']._west(board, the_plot)
        self._stay(board, the_plot)
      else:
        moved = self._east(board, the_plot)
        if moved is not None: # obstructed
          things['P']._west(board, the_plot)

    # move left
    elif (mc - pc == -1) and (mr == pr) and (p_action == 2):
      exiting_room = (self.position == (4, 10))
      if exiting_room == True:
        things['P']._east(board, the_plot)
        self._stay(board, the_plot)
      else:
        moved = self._west(board, the_plot)
        if moved is not None: # obstructed
          things['P']._east(board, the_plot)

    del actions, backdrop  # Unused.


def main(argv=()):
  level = int(argv[1]) if len(argv) > 1 else 0

  # Build the game.
  game = make_game(level)
  # Build the croppers we'll use to scroll around in it, etc.
  croppers = make_croppers(level)

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4,
                       'q': 5, 'Q': 5},
      delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG,
      croppers=croppers)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)
