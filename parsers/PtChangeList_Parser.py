def ReadChangeList(inrpt):
    """Read ECO solution from PT change list, return the size/buffer solution to each pin/cell"""
    size_solution = {} # pin/cell : solution
    buffer_solution = {} # pin/cell : solution
    with open(inrpt, 'r') as infile:
        for line in infile:
            if "size_cell" in line:
                index = line.split()
                size_solution[index[1].replace("{", "").replace("}", "")] = index[2].replace("{", "").replace("}", "")
            elif "insert_buffer" in line:
                index = line.split()
                buffer_solution[index[7].replace("{", "").replace("}", "")] = index[1].replace("{", "").replace("}", "")
    return size_solution, buffer_solution