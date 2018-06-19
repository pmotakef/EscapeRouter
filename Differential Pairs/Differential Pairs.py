# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# key: vertex id, value: vertex coordinate
vertices = {}

# commodity[0] belongs to singles
commodities = []

# key: (vertex1, vertex2), value: edge capacity
edge_capacity = {}

# key: (commodity, (vertex1, vertex2)), value: edge cost
edge_cost = {}

# list of sinks
sinks = []
sink_ids = []

# key: commodity, value: list of vertex coordinates belonging to commodity 
sources = {}

# key: (commodity, vertex), value: Amount of inflow from this vertex
inflow = {}

# key: commodity, value: color
comm_colors = {}

#key: (vertex1, vertex2), value: list of commodities having that edge
edges = {}

use_random_colors = False
    
grid_size = 0
cost_per_length = 1.0
edge_cap = 2.0

sink_node = 0
canvas_width = 800
canvas_height = 600
canvas_margin = 50

# key: (commodity, vertex), value: list of edges entering/leaving the vertex
edgeIn = {}
edgeOut = {}


NON_SHARING_COLLISION = 1
ALL_SHARING = 2
ONE_SIDE_SHARING = 3
NO_COLLISION = 4

SINK_COST_FOR_SINGLES = 0.0001

import gurobipy
from tkinter import Tk, Canvas, mainloop
import random
import itertools

def coord_to_id(i, j):
    return ((j * grid_size) + i) + 1

def id_to_coord(v_id):
    vid = v_id - 1
    i = vid % grid_size
    j = int(vid / grid_size)
    return (i, j)

def add_edge_pair(id1, id2, cost, cap, comms=None):
    if id1 in sink_ids and id2 in sink_ids:
        return
    if not comms:
        comms = commodities
    is_sink = False
    if (id2 == sink_node):
        is_sink = True

    if (id1, id2) not in edge_capacity:
        edge_capacity[(id1, id2)] = cap
    if (id2, id1) not in edge_capacity:
        edge_capacity[(id2, id1)] = cap

    for c in comms:
        if (id1, id2) not in edges:
            edges[(id1, id2)] = []
        edges[(id1, id2)].append(c)
        
        if is_sink and c == 0:
            edge_cost[(c, (id1, id2))] = SINK_COST_FOR_SINGLES
        else:            
            edge_cost[(c, (id1, id2))] = cost
        
        if not is_sink:
            edge_cost[(c, (id2, id1))] = cost
            if (id2, id1) not in edges:
                edges[(id2, id1)] = []
            edges[(id2, id1)].append(c)


def add_single_edge(id1, id2, cost, cap, comm):
    if (id1, id2) not in edge_capacity:
        edge_capacity[(id1, id2)] = cap
    edge_cost[(comm, (id1, id2))] = cost
    
    if (id1, id2) not in edges:
        edges[(id1, id2)] = []
    edges[(id1, id2)].append(comm)

                
def set_inflow(v_id, flow, comms=None):
    if not comms:
        comms = commodities
    for c in comms:
        inflow[(c, v_id)] = flow


def set_edgeInOut(v_id):
    for c in commodities:
        edgeIn[(c, v_id)] = []
        edgeOut[(c, v_id)] = []


def setup_grid():
    n = grid_size
    vertices[sink_node] = (n, n)
    for i in range(0, n):
        for j in range(0, n):
            v_id = coord_to_id(i, j)
            if (i, j) in sinks:
                sink_ids.append(v_id)
            set_edgeInOut(v_id)
            vertices[v_id] = (i, j)
            #set_inflow(v_id, 0)
            if (i > 0):
                add_edge_pair(v_id, coord_to_id(i - 1, j), cost_per_length, edge_cap)
            if (j > 0):
                add_edge_pair(v_id, coord_to_id(i, j - 1), cost_per_length, edge_cap)
            if (i, j) in sinks:
                add_edge_pair(v_id, sink_node, 100.0, edge_cap)
    
    v_counter = 1
    for c in sources:
        edgeIn[(c, sink_node)] = []
        sink_cap = 0
        for (ni, nj) in sources[c]:
            v_id = n * n + v_counter
            i = ni + 0.5
            j = nj + 0.5
            vertices[v_id] = (i, j)
            add_single_edge(v_id, coord_to_id(int(i - 0.5), int(j - 0.5)), cost_per_length / 2.0, edge_cap, c)
            add_single_edge(v_id, coord_to_id(int(i - 0.5), int(j + 0.5)), cost_per_length / 2.0, edge_cap, c)
            add_single_edge(v_id, coord_to_id(int(i + 0.5), int(j - 0.5)), cost_per_length / 2.0, edge_cap, c)
            add_single_edge(v_id, coord_to_id(int(i + 0.5), int(j + 0.5)), cost_per_length / 2.0, edge_cap, c)
            set_inflow(v_id, 1.0, [c])
            edgeOut[(c, v_id)] = []
            v_counter += 1
            sink_cap -= 1
        set_inflow(sink_node, sink_cap, [c])

    if use_random_colors:
        for c in commodities:
            r = lambda: random.randint(0,255)
            col = '#%02X%02X%02X' % (r(),r(),r())
            comm_colors[c] = col


def gurobi_optimize():
    gmodel = gurobipy.Model()
    
    x = {} # Flow on each (commodity, edge)
    y = {} # Binary variable for each (commodity, edge)

    edge_flow = {}

    sink_edges = []
    for c_edge in edge_cost:
        c, edge = c_edge
        u = edge[0] 
        v = edge[1] 

        y[c_edge] = gmodel.addVar(lb=0, ub=1, vtype=gurobipy.GRB.BINARY, name="y" + str(edge) + str(c)) 
        # TODO: double check here for INTEGER or CONTINUOUS
        x[c_edge] = gmodel.addVar(lb=0, vtype=gurobipy.GRB.CONTINUOUS, name="x" + str(edge) + str(c))
        if (c, v) in edgeIn:
            edgeIn[(c, v)] = edgeIn[(c, v)] + [x[c_edge]] 
        if (c, u) in edgeOut:
            edgeOut[(c, u)] = edgeOut[(c, u)] + [x[c_edge]]
            
        if (v == sink_node):
            sink_edges.append(edge)
        
        if (u, v) in edge_flow:
            edge_flow[(u,v)] = edge_flow[(u,v)] + [x[c_edge]]
        elif (v, u) in edge_flow:
            edge_flow[(v,u)] = edge_flow[(v,u)] + [x[c_edge]]
        else:
            edge_flow[(u,v)] = [x[c_edge]]
    
    gmodel.update()
    # Add constraints 
    for c in commodities:
        for v in vertices:
            flow_amount = gurobipy.quicksum(edgeOut.get((c,v), [])) - gurobipy.quicksum(edgeIn.get((c,v), [])) - inflow.get((c, v), 0)
            gmodel.addConstr(flow_amount == 0, name="c%dv%d" % (c, v))
  
    for edge in edge_flow:
        gmodel.addConstr(gurobipy.quicksum(edge_flow[edge]) <= edge_capacity[edge], name='t' + str(edge))

    for c_edge in edge_cost:
        c, edge = c_edge
        gmodel.addConstr(x[c_edge] <= edge_capacity[edge]*y[c_edge], name=str(edge) + str(c))
    
    # Set objective 
    gmodel.setObjective(gurobipy.quicksum((x[c_edge] + edge_cost[c_edge]*y[c_edge]) for c_edge in edge_cost)) 
    
    gmodel.params.LazyConstraints = 1
    gmodel._vars = y
    gmodel.optimize(check_for_crossing)
    

    if gmodel.status == gurobipy.GRB.Status.OPTIMAL:
        print([(yi, y[yi].getAttr('x')) for yi in y if y[yi].getAttr('x') > 0])
        plot_results(x)
  
# =============================================================================
#                               Plotting
# =============================================================================


def get_xy(i, j):
    dx = (canvas_width - 2 * canvas_margin) / (grid_size)
    dy = (canvas_height - 2 * canvas_margin) / (grid_size)
    
    x = canvas_margin + (i * dx)
    y = canvas_margin + (j * dy)
    
    return (x, y)

def plot_results(y_edge):
    master = Tk()

    w = Canvas(master, 
               width=canvas_width,
               height=canvas_height)
    w.pack()
    

    for yi in y_edge:
        line_w = y_edge[yi].getAttr('x')
        cm, (start, end) = yi
        s_i, s_j = vertices[start]
        e_i, e_j = vertices[end]
        sx, sy = get_xy(s_i, s_j)
        ex, ey = get_xy(e_i, e_j)
        fill_col = 'black'
        for c in commodities:
            if inflow.get((c, start), 0) > 0:
                fill_col = comm_colors[c]
        if start in sinks:
            fill_col = 'white'
           
        w.create_oval(sx-4, sy-4, sx+4, sy+4, fill=fill_col)
        if y_edge[yi].getAttr('x') > 0.5:
            fill_col = comm_colors[cm]
            if (end != sink_node):
                w.create_line(sx, sy, ex, ey, fill=fill_col, width=line_w * 2)

    w.update()
    w.postscript(file='sample_pair.ps', colormode='color')
    mainloop() 

# =============================================================================
#                               Crossing
# =============================================================================

def check_for_crossing(model, where):
    if where == gurobipy.GRB.callback.MIPSOL:
        solution = {}
        for var in model._vars:
            cur_sol = model.cbGetSolution(model._vars[var])
            if ( cur_sol > 0.5):
                #   solution {v1: {c: [v2]}}

                c, (v1, v2) = var
                if (v1 != sink_node and v2 != sink_node):
                    if v1 not in solution:
                        solution[v1] = {}
                    if c not in solution[v1]:
                        solution[v1][c] = []
                    solution[v1][c].append(v2)
                    if v2 not in solution:
                        solution[v2] = {}
                    if c not in solution[v2]:
                        solution[v2][c] = []
                    solution[v2][c].append(v1)
                        
        for v in solution:
            if len(solution[v]) > 1:
                collisions = determine_collision(v, solution)
                if len(collisions) > 0:
                    for col in collisions:
                        heads = []
                        tails = []
                        path = []
                        is_head = True
                        for i in range(0, len(col)):
                            if isinstance(col[i], tuple):
                                if is_head:
                                    heads.append(col[i])
                                else:
                                    tails.append(col[i])
                            else:
                                path.append(col[i])
                                is_head = False
                        
                        edge_c = []
                        cs = []
                        for (v, c) in heads:
                            if c not in cs:
                                cs.append(c)
                            e1 = (c, (v, path[0]))
                            e2 = (c, (path[0], v))
                            if e1 in edge_cost:
                                edge_c.append(e1)
                            if e2 in edge_cost:
                                edge_c.append(e2)
                        for (v, c) in tails:
                            if c not in cs:
                                cs.append(c)
                            e1 = (c, (v, path[-1]))
                            e2 = (c, (path[-1], v))
                            if e1 in edge_cost:
                                edge_c.append(e1)
                            if e2 in edge_cost:
                                edge_c.append(e2)
                        for c in set(cs):
                            for i in range(0, len(path) - 1):
                                e1 = (c, (path[i], path[i+1]))
                                e2 = (c, (path[i+1], path[i]))
                                if e1 in edge_cost:
                                    edge_c.append(e1)
                                if e2 in edge_cost:
                                    edge_c.append(e2)
                    

                        expr = gurobipy.quicksum([model._vars[e] for e in edge_c])
                        model.cbLazy(expr - ((len(path) - 1) * 2) <= 3)
                        #print(expr)

                        

def get_neighbor_position(v, n):
    vi, vj = vertices[v]
    ni, nj = vertices[n]
    if ((vi * 10) % 10) == 0:
        pos = {(0, 1): 5, (0.5, 0.5): 4, (1, 0): 3, (0.5, -0.5): 2, (0, -1): 1, (-0.5, -0.5): 8, (-1, 0): 7, (-0.5, 0.5): 6}
    else:
        pos = {(-0.5, 0.5): 6, (0.5, 0.5): 4, (0.5, -0.5): 2, (-0.5, -0.5): 8}
    return pos.get((ni-vi, nj-vj), None)

def get_neighbor_with_offset(v, n, offset):
    ret_pos = get_neighbor_position(v, n)
    if (ret_pos):
        ret_pos += offset
        if (ret_pos > 8):
            ret_pos -= 8
        if (ret_pos < 1):
            ret_pos += 8
    return ret_pos

def determine_collision(vert, sol):
    vc = sol[vert]
    comms = [c for c in vc]
    ret_list = []
 
    for (c1, c2) in itertools.combinations(comms, 2):
        v1 = [(v, c1) for v in vc[c1]]
        v2 = [(v, c2) for v in vc[c2]]
        if len(v1) > 1 and len(v2) > 1:
            v1.extend(v2)
            col_results = x_collision(v1, vert, [c1, c2])
            if NON_SHARING_COLLISION in col_results and len(col_results[NON_SHARING_COLLISION]) > 0:
                for res in col_results[NON_SHARING_COLLISION]:
                    col = res[0:2] + [vert] + res[2:4]
                    ret_list.append(col)
            elif ONE_SIDE_SHARING in col_results and len(col_results[ONE_SIDE_SHARING]) > 0:
                if ALL_SHARING not in col_results or (ALL_SHARING in col_results and len(col_results[ALL_SHARING]) < 1):
                    l_cross = L_crossing(vert, [c1, c2], sol)
                    if len(l_cross) > 0:
                        ret_list.append(l_cross)

    return (ret_list)



def x_collision_for_4(vc_list, ref):
    if len(vc_list) != 4:
        raise ValueError
    pvc = [(get_neighbor_position(v, ref), v, c) for (v, c) in vc_list] if ref else vc_list
    pvc = sorted(pvc, key=lambda tup: tup[0])
    cs = [c for (p,v,c) in pvc]
    ps = list(set([p for (p, v, c) in pvc]))
    if len(ps) <= 2:
        return {ALL_SHARING:[(v,c) for (p,v,c) in pvc]}
    elif len(ps) == 3:
        return {ONE_SIDE_SHARING:[(v,c) for (p,v,c) in pvc]}
    elif cs[1] != cs[2]:
        return {NON_SHARING_COLLISION:[(v,c) for (p,v,c) in pvc]}
    return {NO_COLLISION:[(v,c) for (p,v,c) in pvc]}
    
def x_collision(vc_list, ref, comms):
    vc_list = list(set(vc_list))
    if len(vc_list) < 4:
        return {NO_COLLISION:[vc_list]}
    
    if ref:    
        vc1_list = [(v, c) for (v, c) in vc_list if c == comms[0]]
        vc2_list = [(v, c) for (v, c) in vc_list if c == comms[1]]
    else:
        vc1_list = [(p, v, c) for (p, v, c) in vc_list if c == comms[0]]
        vc2_list = [(p, v, c) for (p, v, c) in vc_list if c == comms[1]]

    if len(vc1_list) < 2 or len(vc2_list) < 2:
        return {NO_COLLISION:vc_list}
    
    results = {}
    for vc1 in itertools.combinations(vc1_list, 2):
        for vc2 in itertools.combinations(vc2_list, 2):
            vc_comb = list(vc1) + (list(vc2))
            res = x_collision_for_4(vc_comb, ref)
            for r in res:
                if not r in results:
                    results[r] = []
                results[r].append(res[r])
    return results
        
    

# [c1, vn1), (c2, vn2), v1, v2, v3, ..., vm-1, vm, (c3, vn3), (c4, vn4)]  Vn(1,2) >===< Vn(3,4)
def check_for_L_crossing(this_vertex, prev_vertex, comms, solution):
    if this_vertex not in solution:
        raise ValueError
    if (prev_vertex is not None):
        continue_path = [v for v in solution[this_vertex][comms[0]] if v in solution[this_vertex][comms[1]] and (v != prev_vertex)]
        if continue_path:
            v_path = [this_vertex]
            temp_sol = get_updated_sol(solution, this_vertex)
            v_path.extend(check_for_L_crossing(continue_path[0], this_vertex, comms, temp_sol))
            return v_path

        next_vertices = []
        if (comms[0] in solution[this_vertex]):
            next_vertices = [v for v in solution[this_vertex][comms[0]] if (v != prev_vertex)]
        if (comms[1] in solution[this_vertex]):
            next_vertices.extend([v for v in solution[this_vertex][comms[1]] if (v != prev_vertex)])
        if not next_vertices:
            raise ValueError

        next_vertices = list(set(next_vertices))

        if len(next_vertices) > 1:
            v_path = [this_vertex]
            for nv in next_vertices:
                if comms[0] in solution[this_vertex] and nv in solution[this_vertex][comms[0]]:
                    v_path.append((comms[0], nv))
                elif comms[1] in solution[this_vertex] and nv in solution[this_vertex][comms[1]]:
                    v_path.append((comms[1], nv))
            return v_path
    else:
        c1 = [v for v in solution[this_vertex][comms[0]]]
        c2 = [v for v in solution[this_vertex][comms[1]]]
        if len(c1) < 2 or len(c2) < 2:
            return []
        sharing_vs = [v for v in c1 if v in c2]
        if sharing_vs:
            sharing_v = sharing_vs[0]
            v1 = [(comms[0], v) for v in c1 if v != sharing_v]
            v2 = [(comms[1], v) for v in c2 if v != sharing_v]
            v_path = v1 + v2
            v_path.append(this_vertex)
            updated_sol = get_updated_sol(solution, this_vertex)
            v_path.extend(check_for_L_crossing(sharing_v, this_vertex, comms, updated_sol))
            return v_path
    return []
            
            

def get_updated_sol(solution, removing_v):
    temp_sol = {}
    for v in solution:
        if v != removing_v:
            temp_sol[v] = solution[v]
    return temp_sol
        
                        
def L_crossing(vertex, comms, solution):
    try:
        cross_path = check_for_L_crossing(vertex, None, comms, solution)
    except ValueError:
        return []
    if len(cross_path) < 4 or (len(set(cross_path)) != len(cross_path)):
        return []
    path_change = []
    heads = []
    tails = []
    is_heads = True
    for i in range(0, len(cross_path)):
        if not isinstance(cross_path[i], tuple):
            path_change.append(cross_path[i])
            is_heads = False
        else:
            if is_heads:
                heads.append(cross_path[i])
            else:
                tails.append(cross_path[i])
    i = 0
    path_offset = 0
    prev_neighbor = 0
    while (i < len(path_change) - 1):
        cur_neighbor = get_neighbor_position(path_change[i], path_change[i+1])
        if i > 0:
            path_offset += (cur_neighbor - prev_neighbor) 
        prev_neighbor = cur_neighbor
        i += 1
    v1 = [(get_neighbor_with_offset(path_change[0], v, 0), v, c) for (c, v) in heads]
    v2 = [(get_neighbor_with_offset(path_change[-1], v, path_offset), v, c) for (c, v) in tails]
    if (len(v1) > 1 and len(v2) > 1):
        v1.extend(v2)
        
        results = x_collision(v1, None, comms)
        if NON_SHARING_COLLISION in results and (len(results[NON_SHARING_COLLISION]) > 0):
            result = []
            hd = [(v, c) for (v, c) in results[NON_SHARING_COLLISION][0] if (c, v) in heads]
            tl = [(v, c) for (v, c) in results[NON_SHARING_COLLISION][0] if (c, v) in tails]
            result = hd + cross_path + tl
            return result
        elif ONE_SIDE_SHARING in results and (len(results[ONE_SIDE_SHARING]) > 0):
            result = []
            hd = [(v, c) for (v, c) in results[ONE_SIDE_SHARING][0] if (c, v) in heads]
            tl = [(v, c) for (v, c) in results[ONE_SIDE_SHARING][0] if (c, v) in tails]
            result = hd + cross_path + tl
            return result
        elif ALL_SHARING in results and (len(results[ALL_SHARING]) > 0):
            result = []
            hd = [(v, c) for (v, c) in results[ALL_SHARING][0] if (c, v) in heads]
            tl = [(v, c) for (v, c) in results[ALL_SHARING][0] if (c, v) in tails]
            result = hd + cross_path + tl
            return result
    return []        


    
if __name__ == '__main__':
    grid_size = 30
    
    canvas_width = 1000
    canvas_height = 1000
    
    sinks = [(i, grid_size - 1) for i in range(0, grid_size - 1)]
    sinks.extend([(i, 0) for i in range(0, grid_size)])
    sinks.extend([(0, i) for i in range(1, grid_size - 1)])
    sinks.extend([(grid_size - 1, i) for i in range(1, grid_size - 1)])
 
    sources[19] = [(14, 6), (14, 7)]
    sources[23] = [(15, 8), (15, 9)]
    sources[22] = [(15, 10), (15, 11)]
    sources[26] = [(16, 9), (16, 11)]
    sources[18] = [(14, 13), (14, 16)]
    sources[21] = [(15, 15), (15, 16)]

    sources[16] = [(13, 12), (13, 15)]
    sources[12] = [(9, 13), (9, 12)]
    sources[6] = [(6, 13), (6, 14)]
    sources[5] = [(6, 24), (6, 25)]
    sources[4] = [(5, 15), (5, 18)]
    sources[3] = [(4, 17), (4, 18)]
    sources[2] = [(3, 18), (3, 21)]
    sources[1] = [(3, 23), (3, 24)]
 
    commodities = [i for i in sources]
    use_random_colors = True
 

# =============================================================================
#     grid_size = 6
#     sinks = [(i, grid_size - 1) for i in range(0, grid_size)]
#     commodities = [1,2,3]    
#     sources[1] = [(1, 1), (3, 1)]
#     sources[2] = [(2, 1), (4, 1)]
#     sources[3] = [(1, 2), (4, 2)] #(3,2)
#     comm_colors = {1: 'red', 2: 'green', 3: 'orange'}
# 
# =============================================================================
    setup_grid()

    gurobi_optimize()

    
    
