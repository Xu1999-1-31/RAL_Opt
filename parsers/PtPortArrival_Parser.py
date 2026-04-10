def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
class port_arrival:
    def __init__(self):
        self.name = ''
        self.arrival = [None, None, None, None] # Max_Rise, Max_Fall, Min_Rise, Min_Fall
    def __repr__(self):
        arrival_repr = f"Max_Rise:{self.arrival[0]}, Max_Fall:{self.arrival[1]}, Min_Rise:{self.arrival[2]}, Min_Fall:{self.arrival[3]}"
        return f"port_arrival(port_name={self.name}, arrival=[{arrival_repr}]"

def Read_PtPortArrival(inrpt):
    """Read port arrival time from PT report, return dict of ports with arrival time on them"""
    ports = {}
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if index[0] == "Information:":
                    continue
                if flag_start and "---" not in index[0]:
                    newport = port_arrival()
                    newport.name = index[4]
                    for i in range(4):
                        if is_float(index[i]):
                            newport.arrival[i] = float(index[i])
                    ports[newport.name] = newport
                if index[0] == "Max_Rise":
                    flag_start = True
    return ports