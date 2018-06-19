from Tkinter import Tk, Canvas, mainloop
import math
import networkx



class Grid:
    def __init__(self, size):
        self.graph = networkx.DiGraph()
        self.grid_size = size
        self.nodes = {}
        self.edges = {}
        self.sources = {}
        self.sinks = []
        self.num_layers = 0
        self.canvas_height = 1300
        self.canvas_width = 1300
        self.canvas_margin = 50
        self.solution = {}
        self.flow_nodes = {}
        self.num_sources = 0

    def coord_to_id(self, i, j, is_source=False):
        n_id = i + (j * self.grid_size)
        if is_source:
            ret_val = (str(n_id) + 's')
        else:
            ret_val = (str(n_id) + 'n')
        return ret_val

    def id_to_coord(self, n_id):
        if n_id == 'sink':
            return (self.grid_size / 2, self.grid_size)
        n = int(n_id[0:-1])
        i = n % self.grid_size
        j = int(n / self.grid_size)
        return (i, j)


    def setup_grid(self):
        self.max_capacity = int(max(1, math.ceil((self.grid_size - 3.0) * (self.grid_size - 3.0) / (4.0 *(self.grid_size - 2.0)))))
        self.create_nodes()
        self.create_edges()

    def create_nodes(self):
        source_size = self.grid_size - 1

        num_cols = int(math.ceil(source_size / 2.0))
        j_offset = int(math.floor(source_size / 2.0))

        for j in range (0, num_cols):
            for i in range(num_cols, num_cols - j - 1, -1):
                if (j % 2 == 0 or i != num_cols - j):
                    if (source_size % 2 == 1 and j % 2 == 1 and i == num_cols ):
                        continue
                    else:
                        sn_id = self.coord_to_id(i, j + j_offset, True)
                        self.graph.add_node(sn_id)
                        self.num_sources += 1

        for j in range (1, num_cols + 1):
            for i in range(num_cols, num_cols - j, -1):
                nn_id = self.coord_to_id(i, j + j_offset)
                self.graph.add_node(nn_id)

        if source_size % 2 == 0:
            for j in range(1, num_cols + 1):
                nn_id = self.coord_to_id(num_cols + 1, j + j_offset)
                self.graph.add_node(nn_id)

        self.graph.add_node('sink')


    def create_edges(self):
        for node in self.graph:
            i, j = self.id_to_coord(node)
            if node[-1] == 'n':
                modes = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                if i == self.grid_size - 1 or j == self.grid_size - 1:
                    modes.remove((i+1, j))
                if i == 0  or j == self.grid_size - 1:
                    modes.remove((i-1, j))
                if j == self.grid_size - 1:
                    modes.remove((i, j+1))
                if j == 0:
                    modes.remove((i, j-1))
                nnodes = [self.coord_to_id(x, y) for (x, y) in modes]
                snodes = [self.coord_to_id(x, y, True) for (x, y) in [(i, j), (i-1, j), (i, j-1), (i-1, j-1)]]
                for n in nnodes:
                    if n in self.graph:
                        self.graph.add_edge(n, node, flow=0, cost=self.grid_size-j)
                for s in snodes:
                    if s in self.graph:
                        self.graph.add_edge(s, node, flow=0, cost=1)
                if j == self.grid_size - 1:
                    self.graph.add_edge(node, 'sink', flow=0, cost=1)
            elif node[-1] == 's' and j == self.grid_size - 2:
                self.graph.add_edge(node, 'sink', flow=0, cost=1)


    def setup_canvas(self, w, h, margin):
        self.canvas_height = h
        self.canvas_width = w
        self.canvas_margin = margin

    def solve(self):
        num_cols = int(math.ceil((self.grid_size - 1.0) / 2.0))
        max_j = self.grid_size - 2

        cur_j = max_j
        cur_i = num_cols
        j_col = max_j
        visited_sources = []
        while len(visited_sources) < self.num_sources:
            node = self.coord_to_id(cur_i, cur_j, True)
            if node in self.graph:
                self.find_shortest_path(node)
                visited_sources.append(node)

            if cur_j == max_j:
                j_col -= 1
                cur_j = j_col
                cur_i = num_cols
            else:
                cur_i -= 1
                cur_j += 1

        #self.sanitize_solution()

    def sanitize_solution(self):
        for node in self.graph:
            if node[-1] == 's':
                continue
            i, j = self.id_to_coord(node)
            nnodes = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            snodes = [(i, j), (i - 1, j), (i, j - 1), (i - 1, j - 1)]
            incoming_edges = {}
            outgoing_edges = {}
            incoming_sources = {}
            for n_c in nnodes:
                n = self.coord_to_id(n_c[0], n_c[1])
                if n in self.graph and (n, node) in self.solution and self.solution[(n, node)] > 0:
                    incoming_edges[n_c] = self.solution[(n, node)]
                if n in self.graph and (node, n) in self.solution and self.solution[(node, n)] > 0:
                    outgoing_edges[n_c] = self.solution[(node, n)]
            for s_c in snodes:
                s = self.coord_to_id(s_c[0], s_c[1], True)
                if s in self.graph and (s, node) in self.solution and self.solution[(s, node)] > 0:
                    incoming_sources[s_c] = self.solution[(s, node)]

            for source in incoming_sources:
                si, sj = source
                neighbor_nodes = [(x, y) for (x, y) in [(si, sj), (si + 1, sj), (si, sj + 1), (si + 1, sj + 1)] if (x, y) != (i, j) and (x, y) in outgoing_edges]
                if len(neighbor_nodes) > 0:
                    exit_node = neighbor_nodes[0]
                    s_id = self.coord_to_id(si, sj, True)
                    e_id = self.coord_to_id(exit_node[0], exit_node[1])
                    self.solution[(s_id, e_id)] = self.solution[(s_id, node)]
                    self.solution[(s_id, node)] = 0
                    self.solution[(node, e_id)] = self.solution[(node, e_id)] - self.solution[(s_id, e_id)]


    def find_shortest_path(self, node):
        try:
            path = networkx.shortest_path(self.graph, source=node, target='sink', weight='cost')
        except:
            print ("error")
            return
        for i in range(0, len(path) - 1):
            edge = (path[i], path[i+1])

            #santize the path
            handled = False
            if path[i][-1] == 'n' and path[i+1][-1] == 'n':
                ni1, nj1 = self.id_to_coord(path[i])
                ni2, nj2 = self.id_to_coord(path[i+1])
                snodes1 = [(ni1, nj1), (ni1 - 1, nj1), (ni1, nj1 - 1), (ni1 - 1, nj1 - 1)]
                snodes2 = [(ni2, nj2), (ni2 - 1, nj2), (ni2, nj2 - 1), (ni2 - 1, nj2 - 1)]
                sneighbors = [(x, y) for (x, y) in snodes1 if (x, y) in snodes2]
                for sn in sneighbors:
                    sn_id = self.coord_to_id(sn[0], sn[1], True)
                    if sn_id not in self.graph:
                        continue
                    if ((sn_id, path[i]) in self.solution and self.solution[(sn_id, path[i])] > 0):
                        self.solution[(sn_id, path[i])] = 0
                        self.solution[(sn_id, path[i+1])] = 1
                        if (sn_id, path[i]) not in self.graph.edges:
                            self.add_back_deleted_node(path[i])
                        self.graph.edges[(sn_id, path[i])]['flow'] = 0
                        self.graph.edges[(sn_id, path[i+1])]['flow'] = 1
                        handled = True
                        break

            if not handled:
                self.graph.edges[edge]['flow'] = self.graph.edges[edge]['flow'] + 1
                self.solution[edge] = self.graph.edges[edge]['flow']
                if self.graph.edges[edge]['flow'] >= self.max_capacity:
                    self.graph.remove_edge(edge[0], edge[1])
            if (self.grid_size - 1) % 2 == 0 and path[i][-1] == 'n' and path[i+1][-1] == 'n':
                si, sj = self.id_to_coord(path[i])
                ei, ej = self.id_to_coord(path[i+1])
                num_cols = int(math.ceil((self.grid_size - 1.0) / 2.0))
                if si == num_cols + 1 and ei == num_cols + 1 and self.graph.edges[edge]['flow'] >= math.ceil(self.max_capacity / 2.0):
                    self.graph.remove_edge(edge[0], edge[1])
            self.check_flow_to_node(path[i+1])


        #print path
        #print self.max_capacity

    def add_back_deleted_node(self, node):
        edges = [(n1, n2) for (n1, n2) in self.solution if n1 == node or n2 == node]
        for (n1, n2) in edges:
            if self.solution[(n1, n2)] < self.max_capacity and (n1, n2) not in self.graph.edges:
                self.graph.add_edge(n1, n2, flow=self.solution[(n1, n2)], cost=1)


    def check_flow_to_node(self, node):
        i, j = self.id_to_coord(node)
        nnodes = [self.coord_to_id(x, y) for (x, y) in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]]
        snodes = [self.coord_to_id(x, y, True) for (x, y) in [(i, j), (i - 1, j), (i, j - 1), (i - 1, j - 1)]]
        flow = 0
        incoming_edges = []
        for n in nnodes:
            if n in self.graph and (n, node) in self.graph.edges:
                flow += self.graph.edges[(n, node)]['flow']
                incoming_edges.append((n, node))
        for s in snodes:
            if s in self.graph and (s, node) in self.graph.edges:
                flow += self.graph.edges[(s, node)]['flow']
                incoming_edges.append((s, node))

        if flow >= int(math.ceil(self.max_capacity * 1.414)):
            #print "exceeded: " + str(flow)
            for edge in incoming_edges:
                self.graph.remove_edge(edge[0], edge[1])

    def get_xy(self, i, j):
        dx = (self.canvas_width - 2 * self.canvas_margin) / (self.grid_size)
        dy = (self.canvas_height - 2 * self.canvas_margin) / (self.grid_size)

        x = self.canvas_margin + (i * dx)
        y = self.canvas_margin + (j * dy)

        return (x, y)

    def plot_results(self):
        grid_x = (self.canvas_width - 2 * self.canvas_margin) / (self.grid_size)
        grid_y = (self.canvas_height - 2 * self.canvas_margin) / (self.grid_size)
        pin_dx = grid_x / 6
        pin_dy = grid_y / 6

        master = Tk()

        w = Canvas(master,
                   width=self.canvas_width,
                   height=self.canvas_height)
        w.pack()


        for node in self.graph:
            i, j = self.id_to_coord(node)
            if node[-1] == 's':
                i += 0.5
                j += 0.5
            nx, ny = self.get_xy(i, j)
            if node[-1] == 's':
                fill_col = 'green'
                w.create_rectangle(nx - pin_dx, ny - pin_dy, nx + pin_dx, ny + pin_dy, fill=fill_col)
            else:
                fill_col = 'black'
                w.create_oval(nx - 2, ny - 2, nx + 2, ny + 2, fill=fill_col)

        self.draw_routes(w)

        w.update()
        w.postscript(file='sol35.ps', colormode='color')
        mainloop()

    def draw_routes(self, canvas_w):
        for node in self.graph:
            if node[-1] == 's':
                continue
            edge_capacities = {}

            i, j = self.id_to_coord(node)
            nnodes = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            snodes = [(i, j), (i - 1, j), (i, j - 1), (i - 1, j - 1)]
            incoming_edges = {}
            outgoing_edges = {}
            incoming_sources = {}

            for n_c in nnodes:
                n = self.coord_to_id(n_c[0], n_c[1])
                if n in self.graph and (n, node) in self.solution and self.solution[(n, node)] > 0:
                    incoming_edges[n_c] = self.solution[(n, node)]
                    edge_capacities[n_c] = self.solution[(n, node)]
                if n in self.graph and (node, n) in self.solution and self.solution[(node, n)] > 0:
                    outgoing_edges[n_c] = self.solution[(node, n)]
                    edge_capacities[n_c] = self.solution[(node, n)]
            for s_c in snodes:
                s = self.coord_to_id(s_c[0], s_c[1], True)
                if s in self.graph and (s, node) in self.solution and self.solution[(s, node)] > 0:
                    incoming_sources[s_c] = self.solution[(s, node)]
                    #edge_capacities[s_c] = self.solution[(s, node)]

            for source in incoming_sources:
                si, sj = source

            max_cap = {'t': 0, 'b': 0, 'r': 0, 'l': 0}
            if (i, j+1) in edge_capacities:
                max_cap['b'] += edge_capacities[(i, j+1)]
            if (i, j-1) in edge_capacities:
                max_cap['t'] += edge_capacities[(i, j-1)]
            if (i-1, j) in edge_capacities:
                max_cap['l'] += edge_capacities[(i-1, j)]
            if (i+1, j) in edge_capacities:
                max_cap['r'] += edge_capacities[(i+1, j)]

            positions = {'t': [], 'b': [], 'l':[], 'r': []}
            for p in positions:
                positions[p] = [0 for k in range(0, max_cap[p])]

            for edge in incoming_edges:
                non_crossings = [ne for ne in outgoing_edges if ne[0] != edge[0] and ne[1] != edge[1]]
                if len(non_crossings) > 0:
                    for nc in non_crossings:
                        while incoming_edges[edge] > 0 and outgoing_edges[nc] > 0:
                            positions = self.draw_line(canvas_w, edge, (i, j), nc, positions, edge_capacities)
                            incoming_edges[edge] = incoming_edges[edge] - 1
                            outgoing_edges[nc] = outgoing_edges[nc] - 1

            out_edges = [e for e in outgoing_edges if outgoing_edges[e] > 0]
            for edge in incoming_sources:
                if len(out_edges) > 0 and incoming_sources[edge] > 0:
                    o_edge = out_edges[0]
                    positions = self.draw_line(canvas_w, edge, (i, j), o_edge, positions, edge_capacities, True)
                    incoming_sources[edge] = incoming_sources[edge] - 1
                    outgoing_edges[o_edge] = outgoing_edges[o_edge] - 1
                    out_edges = [e for e in outgoing_edges if outgoing_edges[e] > 0]

            out_edges = [e for e in outgoing_edges if outgoing_edges[e] > 0]
            in_edges = [e for e in incoming_edges if incoming_edges[e] > 0]
            for l in range(0, min(len(out_edges), len(in_edges))):
                i_edge = in_edges[l]
                o_edge = out_edges[l]
                while incoming_edges[i_edge] > 0 and outgoing_edges[o_edge] > 0:
                    positions = self.draw_line(canvas_w, i_edge, (i, j), o_edge, positions, edge_capacities)
                    incoming_edges[i_edge] = incoming_edges[i_edge] - 1
                    outgoing_edges[o_edge] = outgoing_edges[o_edge] - 1





    def draw_line(self, canvas_w, in_c, c_c, out_c, positions, edge_caps, from_source=False):
        in_orient = self.determine_orientation(c_c[0], c_c[1], in_c[0], in_c[1], from_source)
        out_orient = self.determine_orientation(c_c[0], c_c[1], out_c[0], out_c[1])

        in_side, out_side = self.determine_side(in_orient, out_orient, from_source)
        updated_positions = positions

        if not from_source:
            in_place = self.find_next_empty_pos(positions, in_orient, in_side)
            updated_positions[in_orient][in_place] = 1
            in_coord = self.get_corrected_ij(in_c[0], in_c[1], in_orient, in_place, edge_caps[in_c])
        else:
            in_coord = (in_c[0] + 0.5, in_c[1] + 0.5)

        out_place = self.find_next_empty_pos(positions, out_orient, out_side)
        updated_positions[out_orient][out_place] = 1
        out_coord = self.get_corrected_ij(out_c[0], out_c[1], out_orient, out_place, edge_caps[out_c])



        # in-flow to center
        p2 = None
        if from_source:
            halfway_x = (c_c[0] - in_coord[0]) / 2.0
            halfway_y = (c_c[1] - in_coord[1]) / 2.0
            self.create_line(in_coord[0], in_coord[1], in_coord[0] + halfway_x, in_coord[1] + halfway_y, canvas_w)
            p2 = (in_coord[0] + halfway_x, in_coord[1] + halfway_y)
        else:
            if in_orient == 'l':
                self.create_line(in_coord[0] + 0.25, in_coord[1], c_c[0] - 0.25, in_coord[1], canvas_w)
                p2 = (c_c[0] - 0.25, in_coord[1])
            elif in_orient == 'r':
                self.create_line(in_coord[0] - 0.25, in_coord[1], c_c[0] + 0.25, in_coord[1], canvas_w)
                p2 = (c_c[0] + 0.25, in_coord[1])
            elif in_orient == 't':
                self.create_line(in_coord[0], in_coord[1] + 0.25, in_coord[0], c_c[1] - 0.25, canvas_w)
                p2 = (in_coord[0], c_c[1] - 0.25)
            elif in_orient == 'b':
                self.create_line(in_coord[0], in_coord[1] - 0.25, in_coord[0], c_c[1] + 0.25, canvas_w)
                p2 = (in_coord[0], c_c[1] + 0.25)

        p1 = None
        if out_orient == 'l':
            self.create_line(out_coord[0] + 0.25, out_coord[1], c_c[0] - 0.25, out_coord[1], canvas_w)
            p1 = (c_c[0] - 0.25, out_coord[1])
        elif out_orient == 'r':
            self.create_line(out_coord[0] - 0.25, out_coord[1], c_c[0] + 0.25, out_coord[1], canvas_w)
            p1 = (c_c[0] + 0.25, out_coord[1])
        elif out_orient == 't':
            self.create_line(out_coord[0], out_coord[1] + 0.25, out_coord[0], c_c[1] - 0.25, canvas_w)
            p1 = (out_coord[0], c_c[1] - 0.25)
        elif out_orient == 'b':
            self.create_line(out_coord[0], out_coord[1] - 0.25, out_coord[0], c_c[1] + 0.25, canvas_w)
            p1 = (out_coord[0], c_c[1] + 0.25)

        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])

        if dx == dy:
            self.create_line(p1[0], p1[1], p2[0], p2[1], canvas_w)
        else:
            dx1 = abs(p2[0] - in_coord[0])
            dy1 = abs(p2[1] - in_coord[1])
            dx2 = abs(p1[0] - out_coord[0])
            dy2 = abs(p1[1] - out_coord[1])

            if dy1 == 0 and dy2 == 0:
                center = (p1[0] + p2[0]) / 2.0
                if p2[0] < p1[0]:
                    sign = 1.0
                else:
                    sign = -1.0
                self.create_line(p2[0], p2[1], center - sign * dy / 2.0, p2[1], canvas_w)
                self.create_line(p1[0], p1[1], center + sign * dy / 2.0, p1[1], canvas_w)
                self.create_line(center - sign * dy / 2.0, p2[1], center + sign * dy / 2.0, p1[1], canvas_w)

            elif dx1 == 0 and dx2 == 0:
                center = (p1[1] + p2[1]) / 2.0
                if p2[1] < p1[1]:
                    sign = 1.0
                else:
                    sign = -1.0
                self.create_line(p2[0], p2[1], p2[0], center - sign * dx / 2.0, canvas_w)
                self.create_line(p1[0], p1[1], p1[0], center + sign * dx / 2.0, canvas_w)
                self.create_line(p2[0], center - sign * dx / 2.0, p1[0], center + sign * dx / 2.0, canvas_w)

            elif dx1 == 0 and dy2 == 0:
                if dx < dy:
                    diff = - (p1[0] - p2[0])
                else:
                    diff = (p1[1] - p2[1])
                self.create_line(p2[0], p2[1], p2[0], p1[1] - diff, canvas_w)
                self.create_line(p1[0], p1[1], p2[0] - diff, p1[1], canvas_w)
                self.create_line(p2[0], p1[1] - diff, p2[0] - diff, p1[1], canvas_w)

            elif dx2 == 0 and dy1 == 0:
                if dx < dy:
                    diff = (p1[0] - p2[0])
                else:
                    diff = -(p1[1] - p2[1])
                self.create_line(p1[0], p1[1], p1[0], p2[1] - diff, canvas_w)
                self.create_line(p2[0], p2[1], p1[0] - diff, p2[1], canvas_w)
                self.create_line(p1[0], p2[1] - diff, p1[0] - diff, p2[1], canvas_w)

            elif dy1 == 0 or dy2 == 0:
                center = (p1[0] + p2[0]) / 2.0
                if p2[0] < p1[0]:
                    sign = 1.0
                else:
                    sign = -1.0
                self.create_line(p2[0], p2[1], center - sign * dy / 2.0, p2[1], canvas_w)
                self.create_line(p1[0], p1[1], center + sign * dy / 2.0, p1[1], canvas_w)
                self.create_line(center - sign * dy / 2.0, p2[1], center + sign * dy / 2.0, p1[1], canvas_w)

            elif dx1 == 0 or dx2 == 0:
                center = (p1[1] + p2[1]) / 2.0
                if p2[1] < p1[1]:
                    sign = 1.0
                else:
                    sign = -1.0
                self.create_line(p2[0], p2[1], p2[0], center - sign * dx / 2.0, canvas_w)
                self.create_line(p1[0], p1[1], p1[0], center + sign * dx / 2.0, canvas_w)
                self.create_line(p2[0], center - sign * dx / 2.0, p1[0], center + sign * dx / 2.0, canvas_w)

        return updated_positions

    def create_line(self, si, sj, ei, ej, canvas_w, from_source=False):
        if from_source:
            si += 0.5
            sj += 0.5
        sx, sy = self.get_xy(si, sj)
        ex, ey = self.get_xy(ei, ej)
        canvas_w.create_line(sx, sy, ex, ey, fill='red')

    def get_corrected_ij(self, i, j, orient, pos, max_cap):
        spacing = 0.5 / (max_cap + 1.0)
        ri = i
        rj = j
        if orient == 't' or orient == 'b':
            ri = i - 0.25 + (spacing * (pos + 1.0))
        else:
            rj = j - 0.25 + (spacing * (pos + 1.0))
        return (ri, rj)


    def determine_orientation(self, i1, j1, i2, j2, from_source=False):
        if from_source:
            if i1 == i2:
                return 'br' if j1 == j2 else 'tr'
            return 'bl' if j1 == j2 else 'tl'

        if i1 == i2:
            return 't' if j1 > j2 else 'b'
        return 'l' if i2 < i1 else 'r'

    def sub_coords(self, c1, c2):
        return (c1[0] - c2[0], c1[1] - c2[1])

    def determine_side(self, in_o, out_o, from_source=False):
        if from_source:
            if out_o == 'r' and 't' in in_o:
                return (0, -1)
            if out_o == 'r' and 'b' in in_o:
                return (0, 1)
            if out_o == 'l' and 't' in in_o:
                return (0, -1)
            if out_o == 'l' and 'b' in in_o:
                return (0, 1)
            if out_o == 'b' and 'l' in in_o:
                return (0, -1)
            if out_o == 'b' and 'r' in in_o:
                return (0, 1)
            if out_o == 't' and 'l' in in_o:
                return (0, -1)
            if out_o == 't' and 'r' in in_o:
                return (0, 1)


        choices = {('t', 'l'): (-1, -1), ('t', 'r'): (1, -1), ('b', 'r'): (1, 1), ('b', 'l'): (-1, 1), ('t', 'b'): (-1, -1), ('r', 'l'): (-1, -1)}
        if (in_o, out_o) in choices:
            return choices[(in_o, out_o)]
        choice = choices[(out_o, in_o)]
        return (choice[1], choice[0])

    def find_next_empty_pos(self, positions, orient, side):
        pos = positions[orient]
        start = 0 if side < 0 else len(pos) - 1
        found = False
        while not found:
            if pos[start] == 0:
                return start
            start += -side


if __name__ == '__main__':
    grid = Grid(31)
    grid.setup_grid()
    import timeit
    start = timeit.default_timer()
    grid.solve()
    end = timeit.default_timer()
    print (start - end)
    print (grid.max_capacity)
    grid.plot_results()
