import pandas as pd
import numpy as np

df = pd.read_csv('input', names=['in', 'out'], delimiter=' ')
df['out'] = df['out'].map({'X': 'A', 'Y': 'B', 'Z': 'C'})

# Appreciative of your help yesterday, one Elf gives you an encrypted strategy guide (your puzzle input) that they say will be sure to help you win. "The first column is what your opponent is going to play: A for Rock, B for Paper, and C for Scissors. The second column--" Suddenly, the Elf is called away to help with someone's tent.
#
# The second column, you reason, must be what you should play in response: X for Rock, Y for Paper, and Z for Scissors. Winning every time would be suspicious, so the responses must have been carefully chosen.
#
# The winner of the whole tournament is the player with the highest score. Your total score is the sum of your scores for each round. The score for a single round is the score for the shape you selected (1 for Rock, 2 for Paper, and 3 for Scissors) plus the score for the outcome of the round (0 if you lost, 3 if the round was a draw, and 6 if you won).


def get_score(choices):
    i = choices[0]
    o = choices[1]
    if i == o:
        return 3
    if i == 'A':
        if o == 'B':
            return 6
        if o == 'C':
            return 0
    if i == 'B':
        if o == 'A':
            return 0
        if o == 'C':
            return 6
    if i == 'C':
        if o == 'A':
            return 6
        if o == 'B':
            return 0

# %%


df['out'].map({'A': 1, 'B': 2, 'C': 3}).sum() + df[['in', 'out']].apply(get_score, axis=1).sum()

# %%

# The Elf finishes helping with the tent and sneaks back over to you. "Anyway, the second column says how the round needs to end: X means you need to lose, Y means you need to end the round in a draw, and Z means you need to win. Good luck!"

# %%


def get_choice(choices):
    i = choices[0]
    o = choices[1]
    if o == 'B':  # draw
        return i
    if o == 'A':  # lose
        if i == 'A':
            return 'C'
        if i == 'B':
            return 'A'
        if i == 'C':
            return 'B'
    if o == 'C':  # win
        if i == 'A':
            return 'B'
        if i == 'B':
            return 'C'
        if i == 'C':
            return 'A'


df['out'].map({'A': 0, 'B': 3, 'C': 6}).sum() + df[['in', 'out']].apply(get_choice, axis=1).map({'A': 1, 'B': 2, 'C': 3}).sum()
