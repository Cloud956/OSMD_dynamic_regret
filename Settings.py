from point import *


class Setting:
    def __init__(self):
        self.experiment = 'experiment_1'
        self.cost_mode = 'uniform'
        self.learn_osmd = None
        self.learn_exp = None

    def set_setting(self, setting):
        self.experiment = setting

    def set_cost_mode(self, cost):
        self.cost_mode = cost

    def set_learn_osmd(self, learn):
        self.learn_osmd = learn
    def set_learn_exp(self, learn):
        self.learn_exp = learn

    def give_setting(self):
        return game_settings.get(self.experiment), self.cost_mode, self.learn_osmd, self.learn_exp


class Game:
    def __init__(self, height, length, start, goal, path_len):
        self.h = height
        self.l = length
        self.start = start
        self.goal = goal
        self.path = path_len


small_start = Point(1,1)
small_size = 4
small_goal = Point(3,4)
small_length = 7
big_size =6
big_start=Point(1,1)
big_goal = Point(5,5)
big_length = 10
game_settings = {
    "experiment_1" : (Game(small_size,small_size, small_start,small_goal,small_length)),
    "experiment_2" : (Game(big_size,big_size,big_start,big_goal,big_length))

}

