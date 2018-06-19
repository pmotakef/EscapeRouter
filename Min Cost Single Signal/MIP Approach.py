# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gurobipy
from tkinter import Tk, Canvas, mainloop
import math

# key: (vertex id, vertex side), value: vertex coordinate
vertices = {}
node_ids = []

# key: (vertex1, vertex2), value: edge capacity
edge_capacity = {}

# key: (commodity, (vertex1, vertex2)), value: edge cost
edge_cost = {}

# list of sinks
sinks = []
sink_ids = []
pin_flow = 1.0

pin_constr = {}

# key: commodity, value: list of vertex coordinates belonging to commodity 
sources = {}

# key: (commodity, vertex), value: Amount of inflow from this vertex
inflow = {}

# key: commodity, value: color
comm_colors = {}

#key: (vertex1, vertex2), value: list of commodities having that edge
edges = {}

    
grid_size = 0
cost_per_length = 1.0

sink_node = 0
canvas_width = 800
canvas_height = 600
canvas_margin = 50

# key: vertex, value: list of edges entering/leaving the vertex
edgeIn = {}
edgeOut = {}

TOP = 1
BOTTOM = 2
LEFT = 3
RIGHT = 4
CENTER = 5
PIN = 6

orthogonal_capacity_mulitplier = 1
diagonal_capacity_multiplier = 1.5

SINK_COST_FOR_SINGLES = 0.0001

def coord_to_id(i, j):
    return ((j * grid_size) + i) + 1

def id_to_coord(v_id):
    vid = v_id - 1
    i = vid % grid_size
    j = int(vid / grid_size)
    return (i, j)


def add_dual_edge(edge, cost_multiplier, cap_multiplier):
    i, j = edge
    edge1 = (i, j)
    edge2 = (j, i)
    if not i in edges:
        edges[i] = []
    if not j in edges:
        edges[j] = []
    edges[i].append(edge1)
    edges[j].append(edge2)
    edge_cost[edge1] = cost_per_length * cost_multiplier
    edge_cost[edge2] = cost_per_length * cost_multiplier
    edge_capacity[edge1] = cap_multiplier
    edge_capacity[edge2] = cap_multiplier

def add_single_edge(edge, cost_multiplier, cap_multiplier):
    i, j = edge
    if not i in edges:
        edges[i] = []
    edges[i].append(edge)
    edge_cost[edge] = cost_per_length * cost_multiplier
    edge_capacity[edge] = cap_multiplier


def set_edge_within_node(v_id):
    ortho_cost = 0.5
    diag_cost = 0.35
    top = (v_id, TOP)
    bot = (v_id, BOTTOM)
    left = (v_id, LEFT)
    right = (v_id, RIGHT)
    if top in vertices:
        if bot in vertices:
            add_dual_edge((top, bot), ortho_cost, orthogonal_capacity_mulitplier)
        if right in vertices:
            add_dual_edge((right, top), ortho_cost, diagonal_capacity_multiplier)
        if left in vertices:
            add_dual_edge((left, top), diag_cost, diagonal_capacity_multiplier)
    if bot in vertices:
        if right in vertices:
            add_dual_edge((right, bot), diag_cost, diagonal_capacity_multiplier)
        if left in vertices:
            add_dual_edge((left, bot), diag_cost, diagonal_capacity_multiplier)
    if left in vertices:
        if right in vertices:
            add_dual_edge((left, right), ortho_cost, orthogonal_capacity_mulitplier)

def set_edges_between_nodes(v_id1, v_id2):
    to_sink_cost = 1
    to_node_cost = 1 #0.5
    i1, j1 = id_to_coord(v_id1)
    i2, j2 = id_to_coord(v_id2)
    side1 = TOP
    side2 = BOTTOM
    if i1 > i2:
        side1 = LEFT
        side2 = RIGHT
    elif i1 < i2:
        side1 = RIGHT
        side2 = LEFT
    elif j2 > j1:
        side1 = BOTTOM
        side2 = TOP
    if v_id1 in sink_ids:
        side1 = CENTER
    if v_id2 in sink_ids:
        side2 = CENTER

    v1 = (v_id1, side1)
    v2 = (v_id2, side2)
    if side1 == CENTER and side2 == CENTER:
        return

    if v1 in vertices and v2 in vertices:    
        if side1 == CENTER:
            add_single_edge((v2, v1), to_sink_cost, orthogonal_capacity_mulitplier)
            add_single_edge((v1, (sink_node, CENTER)), SINK_COST_FOR_SINGLES, orthogonal_capacity_mulitplier)
        elif side2 == CENTER:
            add_single_edge((v1, v2), to_sink_cost, orthogonal_capacity_mulitplier)
            add_single_edge((v2, (sink_node, CENTER)), SINK_COST_FOR_SINGLES, orthogonal_capacity_mulitplier)
        else:
            edge = (v1, v2)
            add_dual_edge(edge, to_node_cost, orthogonal_capacity_mulitplier)


def set_pin_edges(i, j):
    v_pin = coord_to_id(i, j)
    node_ul = coord_to_id(i, j)
    node_ur = coord_to_id(i + 1, j)
    node_br = coord_to_id(i + 1, j + 1)
    node_bl = coord_to_id(i, j + 1)
    pin_list = []
    pin_ul = (node_ul, RIGHT) if (node_ul, RIGHT) in vertices else (sink_node, CENTER)
    pin_list.append(pin_ul)
    if (node_ul, BOTTOM) in vertices:
        pin_list.append((node_ul, BOTTOM))

    pin_ur = (node_ur, BOTTOM) if (node_ur, BOTTOM) in vertices else (sink_node, CENTER)
    pin_list.append(pin_ur)
    if (node_ur, LEFT) in vertices:
        pin_list.append((node_ur, LEFT))

    pin_br = (node_br, LEFT) if (node_br, LEFT) in vertices else (sink_node, CENTER)
    pin_list.append(pin_br)
    if (node_br, TOP) in vertices:
        pin_list.append((node_br, TOP))
    
    pin_bl = (node_bl, TOP) if (node_bl, TOP) in vertices else (sink_node, CENTER)
    pin_list.append(pin_bl)
    if (node_bl, RIGHT) in vertices:
        pin_list.append((node_bl, RIGHT))
    
    pin_list = list(set(pin_list))
    for p in pin_list:
        add_single_edge(((v_pin, PIN), p), 0.5, orthogonal_capacity_mulitplier)

    if v_pin not in pin_constr:
        pin_constr[v_pin] = pin_list
    
    #if (node_ul, RIGHT) in pin_list:
    #    cs = []

def set_vertex_prop(vert):
    inflow[vert] = 0
    edgeIn[vert] = []
    edgeOut[vert] = []

def setup_grid():
    vertices[(sink_node, CENTER)] = (grid_size, grid_size)
    edgeIn[(sink_node, CENTER)] = []
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            v_id = coord_to_id(i, j)
            node_ids.append(v_id)
            if ((i, j) not in sinks):
                if (j > 0):
                    vertices[(v_id, TOP)] = (i, j - 0.25)
                    set_vertex_prop((v_id, TOP))
                if (j < grid_size - 1):
                    vertices[(v_id, BOTTOM)] = (i, j + 0.25)
                    set_vertex_prop((v_id, BOTTOM))
                if (i > 0):
                    vertices[(v_id, LEFT)] = (i - 0.25, j)
                    set_vertex_prop((v_id, LEFT))
                if (i < grid_size - 1):
                    vertices[(v_id, RIGHT)] = (i + 0.25, j)
                    set_vertex_prop((v_id, RIGHT))
                
                set_edge_within_node(v_id)
            else:
                vertices[(v_id, CENTER)] = (i, j)
                set_vertex_prop((v_id, CENTER))
                sink_ids.append(v_id)
                
            if (j > 0):
                set_edges_between_nodes(v_id, coord_to_id(i, j - 1))
            if (i > 0):
                set_edges_between_nodes(v_id, coord_to_id(i - 1, j))
            
    sink_capacity = 0
    
    for i in range(0, grid_size -1):
        for j in range(0, grid_size -1):
            v_id = coord_to_id(i, j)
            vertices[(v_id, PIN)] = (i + 0.5, j + 0.5)
            inflow[(v_id, PIN)] = pin_flow
            edgeOut[(v_id, PIN)] = []
            sink_capacity -= pin_flow
            set_pin_edges(i, j)
    
    inflow[(sink_node, CENTER)] = sink_capacity




def gurobi_optimize():
    gmodel = gurobipy.Model()
    
    x = {} # Flow on each (commodity, edge)

    for edge in edge_cost:
        u = edge[0] 
        v = edge[1] 

        x[edge] = gmodel.addVar(lb=0, vtype=gurobipy.GRB.INTEGER, name="x" + str(edge))
        if v in edgeIn:
            edgeIn[v] = edgeIn[v] + [x[edge]] 
        if u in edgeOut:
            edgeOut[u] = edgeOut[u] + [x[edge]]
        
    edge_cap = gmodel.addVar(lb=0, vtype=gurobipy.GRB.INTEGER, name="Cap")
    edge_diag_cap = gmodel.addVar(lb=0, vtype=gurobipy.GRB.INTEGER, name="Cap_diag")
    
    gmodel.addConstr(edge_diag_cap >= ((diagonal_capacity_multiplier * edge_cap) - 0.5))
    gmodel.addConstr(edge_diag_cap <= ((diagonal_capacity_multiplier * edge_cap) + 0.5))
    
    gmodel.update()
    # Add constraints 
    for v in vertices:
        flow_amount = gurobipy.quicksum(edgeOut.get(v, [])) - gurobipy.quicksum(edgeIn.get(v, [])) - inflow.get(v, 0)
        gmodel.addConstr(flow_amount == 0, name="v%s" % str(v))

    for edge in edge_cost:
        gmodel.addConstr(x[edge] <= edge_capacity[edge]*edge_cap, name=str(edge) + 'xc')
       
    for v in node_ids:
        v_t = (v, TOP)
        v_b = (v, BOTTOM)
        v_l = (v, LEFT)
        v_r = (v, RIGHT)
        e_lr = e_tb = e_tl = e_tr = e_bl = e_br = []
        
        if v_l in vertices and v_r in vertices:
            e_lr = [((v_l, v_r)), ((v_r, v_l))]
        if v_t in vertices and v_b in vertices:
            e_tb = [((v_t, v_b)), ((v_b, v_t))]
        if v_l in vertices and v_t in vertices:
            e_tl = [((v_l, v_t)), ((v_t, v_l))]
        if v_r in vertices and v_t in vertices:
            e_tr = [((v_r, v_t)), ((v_t, v_r))]
        if v_l in vertices and v_b in vertices:
            e_bl = [((v_l, v_b)), ((v_b, v_l))]
        if v_r in vertices and v_b in vertices:
            e_br = [((v_r, v_b)), ((v_b, v_r))]
        
        e_cross = e_lr + e_tb
        e_right = e_tr + e_bl
        e_left = e_tl + e_br
        
        diagonal_right_flow = gurobipy.quicksum(x[e] for e in e_right) + (gurobipy.quicksum(diagonal_capacity_multiplier * x[e] for e in e_cross))
        diagonal_left_flow = gurobipy.quicksum(x[e] for e in e_left) + (gurobipy.quicksum(diagonal_capacity_multiplier * x[e] for e in e_cross))
        gmodel.addConstr(diagonal_right_flow <= edge_diag_cap)
        gmodel.addConstr(diagonal_left_flow <= edge_diag_cap)
    
    gmodel.ModelSense = gurobipy.GRB.MINIMIZE

    # Set objective 
    gmodel.setObjective(gurobipy.quicksum((edge_cost[edge]*x[edge]) for edge in edge_cost) + 10000 * edge_cap) 

    gmodel.Params.timeLimit = 600
    gmodel.optimize()
    

    if gmodel.status == gurobipy.GRB.Status.OPTIMAL:
        print("---------------------------------------------------------")
        print(edge_cap)
        print(edge_diag_cap)
        print("---------------------------------------------------------")
        print([(yi, x[yi].getAttr('x')) for yi in x if x[yi].getAttr('x') > 0])
        plot_results(x, edge_cap.getAttr('x'))
    else:
        print("---------------------------------------------------------")
        print(edge_cap)
        print(edge_diag_cap)
        print("---------------------------------------------------------")
        print([(yi, y[yi].getAttr('x')) for yi in y if y[yi].getAttr('x') > 0])
        plot_results(x, edge_cap.getAttr('x'))
        

def get_xy(i, j):
    dx = (canvas_width - 2 * canvas_margin) / (grid_size)
    dy = (canvas_height - 2 * canvas_margin) / (grid_size)
    
    x = canvas_margin + (i * dx)
    y = canvas_margin + (j * dy)
    
    return (x, y)

def plot_results(y_edge, cap):
    grid_x = (canvas_width - 2 * canvas_margin) / (grid_size)
    grid_y = (canvas_height - 2 * canvas_margin) / (grid_size)
    pin_dx = grid_x / 6
    pin_dy = grid_y / 6
    
    
    master = Tk()

    w = Canvas(master, 
               width=canvas_width,
               height=canvas_height)
    w.pack()
    

    for yi in y_edge:
        (start, end) = yi
        if start not in vertices:
            continue
        if end not in vertices:
            continue
        
        ev, es = end
        start_id, start_side = start

      
        s_i, s_j = vertices[start]
        e_i, e_j = vertices[end]
        sx, sy = get_xy(s_i, s_j)
        ex, ey = get_xy(e_i, e_j)
        fill_col = 'black'
        if inflow.get(start, 0) > 0:
            fill_col = 'green'
            w.create_rectangle(sx-pin_dx, sy-pin_dy, sx+pin_dx, sy+pin_dy, fill=fill_col)
        elif start_id in sink_ids:
            fill_col = 'white'
            w.create_oval(sx-4, sy-4, sx+4, sy+4, fill=fill_col)
        else:
            w.create_oval(sx-2, sy-2, sx+2, sy+2, fill=fill_col)

    render_wires(y_edge, cap, w)
    
    w.update()
    w.postscript(file='test.ps', colormode='color')
    mainloop()

def geo_reconstruct_node(flow, cap, sx, sy, ex, ey, canvas_w):
    flow = round(flow + 0.1)
    channel_width = (canvas_width - 2 * canvas_margin) / (grid_size) / 2
    spacing = channel_width / (cap + 1.0)
    
    drawn_line = 0
    offset = float(int(-1.0 * flow / 2.0)) * spacing
    while drawn_line < int(flow):
        x0, y0, x1, y1 = get_line_parallel(sx, sy, ex, ey, offset)
        canvas_w.create_line(x0, y0, x1, y1, fill='red')
        offset += spacing
        drawn_line += 1
    

def find_in_and_out_to_vertex(v_id, solution):
    in_flow = {'l': 0, 'r': 0, 't': 0, 'b': 0}
    internal_flow = {}
    pin_flow = {}
    
    for edge in solution:
        start, end = edge
        s_v, s_s = start
        e_v, e_s = end
        
        if solution[edge].getAttr('x') < 0.4:
            continue
        if e_v == sink_node:
            continue
        
        edge_type = 'tb'
        if (s_s == TOP and e_s == RIGHT) or (e_s == TOP and s_s == RIGHT):
            edge_type = 'drt'
        elif (s_s == BOTTOM and e_s == LEFT) or (e_s == BOTTOM and s_s == LEFT):
            edge_type = 'dlb'
        elif (s_s == TOP and e_s == LEFT) or (e_s == TOP and s_s == LEFT):
            edge_type = 'dlt'
        elif (s_s == BOTTOM and e_s == RIGHT) or (e_s == BOTTOM and s_s == RIGHT):
            edge_type = 'drb'
        elif (s_s == RIGHT and e_s == LEFT) or (e_s == RIGHT and s_s == LEFT):
            edge_type = 'lr'


           
        if s_v == v_id and e_v == v_id and s_s != PIN:
            if edge_type not in internal_flow:
                internal_flow[edge_type] = 0
            internal_flow[edge_type] += solution[edge].getAttr('x')
        elif s_v == v_id or e_v == v_id:
            if s_s == PIN:
                if e_v != v_id:
                    continue
                si, sj = id_to_coord(s_v)
                ei, ej = id_to_coord(e_v)
                if abs(ei - si) == 1 and abs(ej - sj) == 1:
                    if e_s == TOP:
                        edge_type = 't'
                        pin_wire = '0t'
                    else:
                        edge_type = 'l'
                        pin_wire = '0l'
                elif abs(ei - si) == 1:
                    if e_s == BOTTOM:
                        edge_type = 'b'
                        pin_wire = '0b'
                    else:
                        edge_type = 'l'
                        pin_wire = '1l'
                elif abs(ej - sj) == 1:
                    if e_s == TOP:
                        edge_type = 't'
                        pin_wire = '1t'
                    else:
                        edge_type = 'r'
                        pin_wire = '0r'
                else:
                    if e_s == BOTTOM:
                        edge_type = 'b'
                        pin_wire = '1b'
                    else:
                        edge_type = 'r'
                        pin_wire = '1r'
                pin_flow[s_v] = pin_wire

            elif s_v == v_id:
                if e_v == s_v + 1:
                    edge_type = 'r'
                elif e_v == s_v - 1:
                    edge_type = 'l'
                elif e_v > s_v:
                    edge_type = 'b'
                else:
                    edge_type = 't'
            else: # e_v == v_id:
                if e_v == s_v + 1:
                    edge_type = 'l'
                elif e_v == s_v - 1:
                    edge_type = 'r'
                elif e_v < s_v:
                    edge_type = 'b'
                else:
                    edge_type = 't'
            
            if edge_type not in in_flow:
                in_flow[edge_type] = 0
            
            in_flow[edge_type] += solution[edge].getAttr('x')
        
    return (in_flow, internal_flow, pin_flow)
    
def determine_vertex_geo(v_id, solution, max_cap):
    flow, internal_flow, pin_flow = find_in_and_out_to_vertex(v_id, solution)
    
    positions = {'l': [], 'r': [], 't': [], 'b': []}
    used_positions = {'l': [], 'r': [], 't': [], 'b': []}
    ret_positions = {'l': [], 'r': [], 't': [], 'b': []}
    pin_positions = []
    
    for p in flow:
        p_flow = int(round(flow[p] + 0.1))
        offset = int(round((max_cap - p_flow) / 2.0))
        positions[p] = [str(f + offset)+p for f in (range(0, p_flow))]
        ret_positions[p] = [(v_id, str(f + offset)+p) for f in (range(0, p_flow))]
    
    def get_next_pos(side, direction=1):
        start = 0
        end = len(positions[side])
        if (direction < 0):
            end = -1
            start = len(positions[side]) - 1

        for i in range(start, end, direction):
            p = positions[side][i]
            if p not in used_positions[side]:
                return p
        return None
    
    flow_edges = []
    
    for pf in pin_flow:
        f = pin_flow[pf]
        sel = f[-1]
        dc = 1 if f[0] == '0' else -1
        c = get_next_pos(sel, dc)
        flow_edges.append(((pf, 'pin'), (v_id, c)))
        pin_positions.append(pf)
    
    selections = {'drt': ('t', 'r'), 'dlb': ('b', 'l'), 'dlt': ('t', 'l'), 'drb': ('b', 'r'), 'lr': ('l', 'r'), 'tb': ('t', 'b'),}
    directions = {'drt': (-1 ,  1 ), 'dlb': (1 ,  -1 ), 'dlt': ( 1 ,  1 ), 'drb': (-1 , -1 ), 'lr': (1  ,  1 ), 'tb': (1  ,  1 ),}
    for pos in internal_flow:
        if pos[0] != 'd':
            continue
        nums = int(round(internal_flow[pos] + 0.1))
        sel1, sel2 = selections[pos]
        dir1, dir2 = directions[pos]
        for f in range(0, nums):
            c1 = get_next_pos(sel1, dir1)
            c2 = get_next_pos(sel2, dir2)
            flow_edges.append(((v_id, c1), (v_id, c2)))
            used_positions[sel1].append(c1)
            used_positions[sel2].append(c2)

    for pos in internal_flow:
        if pos[0] == 'd':
            continue
        nums = int(round(internal_flow[pos] + 0.1))
        sel1, sel2 = selections[pos]
        dir1, dir2 = directions[pos]
        for f in range(0, nums):
            c1 = get_next_pos(sel1, dir1)
            c2 = get_next_pos(sel2, dir2)
            flow_edges.append(((v_id, c1), (v_id, c2)))
            used_positions[sel1].append(c1)
            used_positions[sel2].append(c2)
        
    return (flow_edges, ret_positions, pin_positions)
    
    
   

def create_all_edges(solution, max_cap):
    spacing = 0.5 / (max_cap + 1.0)
    
    flow_edges = []
    flow_positions = {}
    
    pos_coords = {}
    
    for v_id in node_ids:
        if v_id == 12:
            print('41')
        es, ps, pin_pos = determine_vertex_geo(v_id, solution, max_cap)
        flow_edges.extend(es)
        flow_positions[v_id] = ps
        
        i, j = id_to_coord(v_id)
        
        for vert in pin_pos:
            pi, pj = id_to_coord(vert)
            pos_coords[(vert, 'pin')] = (pi + 0.5, pj + 0.5)
        
        for side in ps:
            p_list = ps[side]
            for p_tup in p_list:
                vd, p = p_tup
                if p[-1] == 't' or p[-1] == 'b':
                    if p[-1] == 't':
                        y = j - 0.2
                    else:
                        y = j + 0.2
                    x = i - 0.25 + (spacing * (int(p[0:-1]) + 1))
                    pos_coords[(v_id, p)] = (x, y)
                elif p[-1] == 'r' or p[-1] == 'l':
                    if p[-1] == 'l':
                        x = i - 0.2
                    else:
                        x = i + 0.2
                    y = j - 0.25 + (spacing * (int(p[0:-1]) + 1))
                    pos_coords[(v_id, p)] = (x, y)
        
        
    all_pins = [((v1, s1), (v2, s2)) for ((v1, s1), (v2, s2)) in flow_edges if s1 == 'pin']
    
    
    for v_id in node_ids:
        ni, nj = id_to_coord(v_id)
        v_pin = coord_to_id(ni - 1, nj - 1)
        if ni > 0:
            v_prev = coord_to_id(ni - 1, nj)
            pos_cur_v = flow_positions[v_id]['l']
            pos_pre_v = flow_positions[v_prev]['r']
            nums = min(len(pos_cur_v), len(pos_pre_v))
            for i in range(0, nums):
                k = i
                l = i
                pin_cur =[p for p in all_pins if p[0] == (v_pin, 'pin') and p[1][0] == v_id]
                pin_prev =[p for p in all_pins if p[0] == (v_pin, 'pin') and p[1][0] == v_prev]
                if len(pin_cur) > 0 and pin_cur[0][1][1][-1] == 'l':
                    if int(pin_cur[0][1][1][0:-1]) <= int(pos_cur_v[i][1][0:-1]):
                        l = i + 1
                elif len(pin_prev) > 0 and pin_prev[0][1][1][-1] == 'r':
                    if int(pin_prev[0][1][1][0:-1]) <= int(pos_pre_v[i][1][0:-1]):
                        k = i + 1
                try:
                    edge = (pos_cur_v[l], pos_pre_v[k])
                except:
                    pass
                flow_edges.append(edge)
        if nj > 0:
            v_prev = coord_to_id(ni, nj - 1)
            pos_cur_v = flow_positions[v_id]['t']
            pos_pre_v = flow_positions[v_prev]['b']
            nums = min(len(pos_cur_v), len(pos_pre_v))
            for i in range(0, nums):
                k = i
                l = i
                pin_cur =[p for p in all_pins if p[0] == (v_pin, 'pin') and p[1][0] == v_id]
                pin_prev =[p for p in all_pins if p[0] == (v_pin, 'pin') and p[1][0] == v_prev]
                if len(pin_cur) > 0 and pin_cur[0][1][1][-1] == 't':
                    if int(pin_cur[0][1][1][0:-1]) <= int(pos_cur_v[i][1][0:-1]):
                        l = i + 1
                elif len(pin_cur) > 0 and pin_cur[0][1][1][-1] == 'b':
                    if int(pin_cur[0][1][1][0:-1]) <= int(pos_cur_v[i][1][0:-1]):
                        k = i + 1
                edge = (pos_cur_v[l], pos_pre_v[k])
                flow_edges.append(edge)
            
    return (pos_coords, flow_edges)

def render_wires(solution, max_cap, canvas_w):
    EPSILON = 0.0001
    coords, f_edges = create_all_edges(solution, max_cap)
    for edge in f_edges:
        s, e  = edge
        try:
            sx, sy = get_xy(coords[s][0], coords[s][1])
            ex, ey = get_xy(coords[e][0], coords[e][1])
        except KeyError:
            print ('Error: ' + str(edge))
            continue
        dx = abs(sx - ex)
        dy = abs(sy - ey)
        if dx < EPSILON or dy < EPSILON:
            canvas_w.create_line(sx, sy, ex, ey, fill='red')
        else:
            if dy < dx:
                center = dx / 2.0 + min(sx, ex)
                p1 = center - (dy / 2.0)
                p2 = center + (dy / 2.0)
                if sx < ex:
                    canvas_w.create_line(sx, sy, p1, sy, fill='red')
                    canvas_w.create_line(p1, sy, p2, ey, fill='red')
                    canvas_w.create_line(p2, ey, ex, ey, fill='red')
                else:
                    canvas_w.create_line(ex, ey, p1, ey, fill='red')
                    canvas_w.create_line(p1, ey, p2, sy, fill='red')
                    canvas_w.create_line(p2, sy, sx, sy, fill='red')
            else:
                center = dy / 2.0 + min(sy, ey)
                p1 = center - (dx / 2.0)
                p2 = center + (dx / 2.0)
                if sy < ey:
                    canvas_w.create_line(sx, sy, sx, p1, fill='red')
                    canvas_w.create_line(sx, p1, ex, p2, fill='red')
                    canvas_w.create_line(ex, p2, ex, ey, fill='red')
                else:
                    canvas_w.create_line(ex, ey, ex, p1, fill='red')
                    canvas_w.create_line(ex, p1, sx, p2, fill='red')
                    canvas_w.create_line(sx, p2, sx, sy, fill='red')
        

def get_line_parallel(x0, y0, x1, y1, d):
    x = x1 - x0
    y = y1 - y0
    if x == 0:
        return (x0 + d, y0, x0 + d, y1)
    if y == 0:
        return (x0, y0 + d, x1, y0 + d)
    
    length = math.sqrt((x*x) + (y*y))
    x0n = x0 + (d * (-y) / length)
    y0n = y0 + (d * x / length)
    x1n = x1 + (d * (-y) / length)
    y1n = y1 + (d * x / length)
    return (x0n, y0n, x1n, y1n)
    



if __name__ == '__main__':
    grid_size = 26
    
    canvas_width = 1000
    canvas_height = 1000
    
    sinks = [(i, grid_size - 1) for i in range(0, grid_size)]
    sinks.extend([(grid_size - 1, i) for i in range(0, grid_size - 1)])
    sinks.extend([(0, i) for i in range(0, grid_size - 1)])
    sinks.extend([(i, 0) for i in range(1, grid_size - 1)])
    
    setup_grid()
    
    #print (vertices)
    #print (edge_cost)
    gurobi_optimize()
    
    
    
    
    
    
    