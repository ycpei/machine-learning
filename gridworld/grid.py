
class Model:
    def utility(state):
        return

class Gridworld(Model):
    PCH = '.' #passage char
    TCH = '^' #trap char
    MCH = '$' #money / treasure char
    ICH = ' ' #impasse char
    PRW = -.04 #passage reward
    TRW = -1 #trap reward
    MRW = 1 #treasure reward
    SWT = {PCH: PRW, TCH: TRW, MCH: MRW} #state-reward table
    ACTS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    D, R, U, L = 0, 1, 2, 3

    def __init__(mfname, sideprob=.1):
        with open(mfname, 'r') as plan:
            self.grid = plan.readlines()
        for x in len(self.grid):
            self.grid[x] = ICH + self.grid[x] + ICH
        self.states = [(x, y) for y in self.grid[x] for x in range(len(self.grid))]
        self.sideprob = sideprob
        gen_transprobtable()

    def gen_transprobtable():
        self.transprobtable = [[[{} for _ in range(4)] for _ in grid[x]] for x in range(len(grid))]
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                if grid[x][y] == ICH:
                    continue
                for action in range(4):
                    dx, dy = ACTS[action]
                    dest = {(x, y): 0, (x + dx, y + dy): 1 - 2 * sideprob, (x + dy, y + dx): sideprob}
                    if dx == 0:
                        dest[(x - dy, y + dx)] = sideprob
                    else:
                        dest[(x + dy, y - dx)] = sideprob
                    for (x_, y_), v in dest:
                        if grid[x_][y_] == ICH:
                            dest[(x, y)] += v
                            dest[(x_, y_)] = 0
                    self.transprobtable[x][y][i] = {k: v for k, v in dest if v}

        def reward(state):
            x, y = state
            return SWT[grid[x][y]]

        def transprob(state, action):
            x, y = state
            return transprobtable([x][y][action], {})
        
        def states():
            return self.states;

class Utility:
    def __init__(model):
        self.model = model
        self.uarr = {state: model.reward(state) for state in model.states()}

    def iterate():
        

    def iterate(policy):
        return

class Policy:
    def iterate():
        return
