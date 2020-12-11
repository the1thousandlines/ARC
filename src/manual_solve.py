#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.


def attaching(p,points):
#     print(p,points,np.abs(np.array([abs(p[0]-pi[0]) + abs(p[1]-pi[1]) for pi in points])))
    return np.any(np.array([abs(p[0]-pi[0]) + abs(p[1]-pi[1]) for pi in points])<=1)


def make_raltive(anchor, fills):
    ret = [(f[0]-anchor[0],f[1]-anchor[1]) for f in fills]
    return ret

def findClusterPair(cluster,clusters):
    size = len(cluster)
    relevants = [c for c in clusters if len(c)==size]
    return relevants

def wouldFit(c1,c2,xdist,ydist):
    size = np.sqrt(len(c1))
    # print("trying",c1[-1][0]-c2[-1][0],c1[-1][1]-c2[-1][1],"s",xdist,ydist)
    return c1[-1][0]-c2[-1][0] == size*xdist and c1[-1][1]-c2[-1][1]==size*ydist

def fill_raltives_with_size(rels, anchor, size,y):
    print(anchor,size)
    for off in rels:
        # print("off",off)
        start = (anchor[0]+off[0]*size, anchor[1]+off[1]*size)
        # print("ss",start)
        for xi in range(size):
            for yi in range(size):
#                 print("filling",)
#                 print(start[0]-xi,start[1]-yi)
                if start[1]-yi>=0 and start[0]-xi>=0:
                    y[start[1]-yi] [start[0]-xi]= 1
    return y

# chosen 1
def solve_447fd412(x):
    """
    Problem:
        problem 447fd412 is a problem, in which a small blueprint of a shape is given
        with one or two 'anchor' points. The task is to replicate/project theses shapes onto 
        the other anchors in X, taking into consideration that their size might have changed.
        If a replica would hit a broder of the field, the rest is supposed to be cut off.

    Algorithm:
        The algorithm work as follows:
            1. identify the already drawn shape (only ones in X)
            2. identify attaching anchors (attaching 2's) they shall be the 'pins' to pin the shape on
            3. if there are 2 anchors, identify the x and y distance
                3.1 compute the relative position of all the fill points to the upper left anchor 
            4. identify all possible other clusters that are anchors (clusters of 2's)
            5. order them, from now on only the size and bottom right cornor will be considered
            6. choose a new anchor, and determine size
            7. if there were multiple pins, find other candidates (same sized anchors)
            8. remove those that are in the right position to fit from the anchor list
            9. choose the upper left most anchor
            10. fill in the shape, using the size of the anchor and its bottom right cornor
                10.1 ignore part that would be outside the field
    Solved Grids:
        All girds were solved correctly
    """
    calibs=[]
    fills=[]
    for xi in range(x.shape[1]):
        for yi in range(x.shape[0]):
            if x[yi,xi]==1:
                fills.append((xi,yi))
            if x[yi,xi]==2:
                calibs.append((xi,yi))


    pins = [p for p in calibs if attaching(p,fills)]

    pins.sort()

    relatives = make_raltive(pins[0], fills)
    if len(pins)==2:

        xdist = abs(pins[0][0]-pins[1][0])
        ydist = abs(pins[0][1]-pins[1][1])

        clusters = []
        calibC = calibs.copy()
        for p in pins:
            calibC.remove(p)
        while len(calibC)>0:
            cluster = [calibC[0]]
            calibC.remove(calibC[0])
            changed = True
            while changed:
                changed = False
                clusterN = [p for p in calibC if attaching(p,cluster)]
                for p in clusterN:
                    calibC.remove(p)
                    cluster.append(p)
                    changed=True
            cluster.sort()
            clusters.append(cluster)

        y = x.copy()
        while len(clusters)>0:
            current = clusters.pop()
            brcorner = current[-1]
            candidates = findClusterPair(current,clusters)
            # print(current[-1],candidates)
            for can in candidates:
                if wouldFit(current,can,xdist,ydist):
                    clusters.remove(can)
                    # print(can)
                    if can[-1][0]<brcorner[0] or can[-1][1]<brcorner[1]:
                        brcorner=can[-1]
            # print("corner",brcorner)

            y = fill_raltives_with_size(relatives,brcorner, \
                                    int(np.sqrt(len(current))) \
                                    ,y)
    else: 
        clusters = []
        calibC = calibs.copy()
        y = x.copy()
        for p in pins:
            calibC.remove(p)
        while len(calibC)>0:
            cluster = [calibC[0]]
            calibC.remove(calibC[0])
            changed = True
            while changed:
                changed = False
                clusterN = [p for p in calibC if attaching(p,cluster)]
                for p in clusterN:
                    calibC.remove(p)
                    cluster.append(p)
                    changed=True
            cluster.sort()
            clusters.append(cluster)
            
        while len(clusters)>0:
            current = clusters.pop()
            brcorner = current[-1]
            y = fill_raltives_with_size(relatives,brcorner, \
                                    int(np.sqrt(len(current))) \
                                    ,y)
    return y

# def solve_6d58a25d(x):
#     return x

# def solve_05269061(x):
#     return x

def solve_c3e719e8(case):
    """
    Problem:
        In problem 6d58a25d a 3x3 grid is given. The most common colour in the grid represents which section of a 9x9 grid the given
        grid is placed (think of a sudoku grid and matchning the colour to a 3x3 box within the sudoku grid).

    Algorithm:
        The algorithm work as follows:
            1. Identify the most common colour
            2. Map the indexes of the colour to the 9x9 grid.
            3. Place the 3x3 grid in the 9x9 grid based on these indexes
            4. Return the solution
            
    Solved Grids:
        All girds were solved correctly
    """
    old = np.array(case)
    c = Counter([x for y in case for x in y])
    n = max(c, key=lambda x:c[x])
    new = np.zeros((9,9))
    for i,line in enumerate(case):
        for j,num in enumerate(line):
            if num == n:
                new[3*i:(i+1)*3,3*j:3*(j+1)] = old
            else:
                continue
    return new.astype(int).tolist()


def solve_6d58a25d(inp):
    """
    Problem:
        In problem 6d58a25d a of a shape is given (with colour a). The task is to check 'under' the shape and
        if a block of colour b (no reason b cant be a but in the examples and test such a situation does not arise)
        is found then a 'line' is drawn from the bottom of the grid striaght up through the block and to the base of
        the shape.

    Algorithm:
        The algorithm work as follows:
            1. Using a moving window match the shape
            2. Identify points beneath the shape, store their column and colour
            3. Draw the 'line' through each point found.
            4. Return the solution
            
    Solved Grids:
        All girds were solved correctly
    """
    inp = np.array(inp)
    shape = np.array([[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,0,1,1,0],[1,0,0,0,0,0,1]])
    tot = np.count_nonzero(shape)
    y = shape.shape[0]
    x = shape.shape[1]
    for j in range(inp.shape[0]):
        for i in range(inp.shape[1]):
            try:
                A = np.multiply(inp[j:j+y,i:i+x], shape)
            except:
                continue
            if (np.count_nonzero(A) == tot):
                col_idxs = list(range(i,i+x))
                height = j+2
                box = inp[j:j+y,i:i+x]
                break
        else:
            continue
        break
            
    colour = box.max()

    colourings = []
    for i,x in enumerate(inp[height:,col_idxs].T):
        for y in x:
            if y not in [0,colour]:
                colourings.append((y,min(col_idxs)+i))
                
    for tup in colourings:
        flag = 0
        for i in range(inp[:,tup[1]].shape[0]):
            if inp[i,tup[1]] == colour:
                flag = 1
            if flag and not (inp[i,tup[1]] == colour):
                inp[i,tup[1]] = tup[0]
    return inp


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

