# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
import os, sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageChops
from time import time

from Solve2by2 import solve2by2
from Solve3by3 import solve3by3
from Solve3by3_till_C import solve3by3_till_C


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):

        type  = problem.problemType
        ans = -1
        # print(problem.problemSetName)


        if type == "2x2":
            ans = solve2by2(problem)
        elif type == "3x3":
            ans = solve3by3(problem) if 'Problems C' not in problem.problemSetName else solve3by3_till_C(problem) 
        else:
            ans =-1
        return ans

