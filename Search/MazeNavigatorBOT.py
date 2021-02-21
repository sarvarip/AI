class Node:
    def __init__(self, position:(), parent:()):
        self.position = position
        self.parent = parent
        self.c = 0 #cost to get here from parent
        self.h = 0 #heuristic
        self.tc = 0 #total cost
        self.steps = 1 #including itself
        
    def __eq__(self, other): #needed for when I test (neighbor in closed_list)
        return self.position == other.position
    
def extend_path(pos,arr_in,straight_cost,diag_cost):
    (x,y,z) = pos
    nbours = []
    if arr_in == None:
        return []
    if 1 in arr_in:
        nbours.append((x+1,y,z,straight_cost))
    if 2 in arr_in:
        nbours.append((x-1,y,z,straight_cost))
    if 3 in arr_in:
        nbours.append((x,y+1,z,straight_cost))
    if 4 in arr_in:
        nbours.append((x,y-1,z,straight_cost))
    if 5 in arr_in:
        nbours.append((x,y,z+1,straight_cost))
    if 6 in arr_in:
        nbours.append((x,y,z-1,straight_cost))
    if 7 in arr_in:
        nbours.append((x+1,y+1,z,diag_cost))
    if 8 in arr_in:
        nbours.append((x+1,y-1,z,diag_cost))
    if 9 in arr_in: 
        nbours.append((x-1,y+1,z,diag_cost))
    if 10 in arr_in:
        nbours.append((x-1,y-1,z,diag_cost))
    if 11 in arr_in:
        nbours.append((x+1,y,z+1,diag_cost))
    if 12 in arr_in:
        nbours.append((x+1,y,z-1,diag_cost))
    if 13 in arr_in: 
        nbours.append((x-1,y,z+1,diag_cost))
    if 14 in arr_in:
        nbours.append((x-1,y,z-1,diag_cost))
    if 15 in arr_in:
        nbours.append((x,y+1,z+1,diag_cost))
    if 16 in arr_in:
        nbours.append((x,y+1,z-1,diag_cost))
    if 17 in arr_in:
        nbours.append((x,y-1,z+1,diag_cost))
    if 18 in arr_in:
        nbours.append((x,y-1,z-1,diag_cost))
    return nbours
    
grid_actions = {}
with open('input.txt', 'r') as f:
    #First line
    line = f.readline()
    algo = line.strip().split(' ')[0] #split(' ')[0] to be on the safe side 
    
    if algo == "BFS":
        straight_cost = 1
        diag_cost = 1
    if algo != "BFS":
        straight_cost = 10
        diag_cost = 14
    
    #Second line
    line = f.readline()
    linearr = line.strip().split(' ')
    max_x = int(linearr[0])
    max_y = int(linearr[1])
    max_z = int(linearr[2])
    #Third line
    line = f.readline()
    linearr = line.strip().split(' ')
    start_x = int(linearr[0])
    start_y = int(linearr[1])
    start_z = int(linearr[2])
    #Fourth line
    line = f.readline()
    linearr = line.strip().split(' ')
    goal_x = int(linearr[0])
    goal_y = int(linearr[1])
    goal_z = int(linearr[2])
    #Fifth line
    line = f.readline()
    numgrids = line.strip().split(' ')[0]
    #Further lines
    for i in range(int(numgrids)):
        line = f.readline()
        elem = line.strip().split(' ')
        elems = [int(e) for e in elem]
        (pos_x,pos_y,pos_z) = elems[:3]
        actions = elems[3:]
        edges = extend_path((pos_x,pos_y,pos_z),actions,straight_cost,diag_cost)
        grid_actions[(pos_x,pos_y,pos_z)] = edges
        
opened_list = []
opened_set = set()
closed_set = set()
start = Node((start_x,start_y,start_z), None)
opened_list.append(start)
opened_set.add(start.position)
goal = Node((goal_x,goal_y,goal_z), None)
solution = []

# Loop until the open list is empty
while len(opened_list) > 0:
    # print([(o.position, o.tc + o.h) for o in opened_list])
    current_node = opened_list.pop(0)
    # Add the current node to the closed list
    closed_set.add(current_node.position)

    # Check if we have reached the goal, return the path
    if current_node == goal:
        path = []
        while current_node != start:
            path.append(current_node)
            current_node = current_node.parent
        #path.append(start) 
        # Return reversed path
        path.append(start)
        solution = path[::-1]
        break
        
    (x,y,z) = current_node.position
    # Get neighbors
    pos_neighbors = grid_actions.get((x,y,z))
    # Loop neighbors
    if not pos_neighbors: #list is empty
        continue 
    for pos_nbor in pos_neighbors:
        # Create a neighbor node
        (x,y,z,cost) = pos_nbor
        neighbor = Node((x,y,z), current_node)
        # Check if the neighbor is in the closed list
        if (neighbor.position in closed_set):
            continue
        if (neighbor.position[0] >= max_x or neighbor.position[1] >= max_y or neighbor.position[2] >= max_z):
            continue
        neighbor.c = cost
        neighbor.tc = current_node.tc + cost
        neighbor.steps = current_node.steps + 1
        if (neighbor.position not in opened_set):
            opened_list.append(neighbor)
            opened_set.add(neighbor.position)
                
# if not solution:
#     solution = "FAIL"
#     print(solution)
# else:
#     print(solution[-1].tc)
#     print(solution[-1].steps)
#     for sol in solution:
#         print(" ".join(str(x) for x in sol.position) + " " + str(sol.c))

with open('output.txt', 'w+') as f:
    if not solution:
        solution = "FAIL"
        f.write(solution)
    else:
        f.write(str(solution[-1].tc) + '\n')
        f.write(str(solution[-1].steps) + '\n')
        for sol in solution:
            f.write(" ".join(str(x) for x in sol.position) + " " + str(sol.c) + '\n')
