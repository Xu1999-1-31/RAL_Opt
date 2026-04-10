import linecache
import re

class Net:
    def __init__(self):
        self.totalCap_= 0
        self.resistance = 0
        self.name = ''
        self.inpins = []
        self.inpin_caps = []
        self.isinPinPIPO = []
        self.outpins = []
        self.outpin_caps = []
        self.isoutPinPIPO = []
    def __repr__(self):
        inpins_repr = ', '.join([f'{pin}' for pin in self.inpins])
        outpins_repr = ', '.join([f'{pin}' for pin in self.outpins])
        return f"Net(name='{self.name}', totalCap='{self.totalCap:.8f}', \ninpins={{ {inpins_repr} }}, \noutpins={{ {outpins_repr} }}\n)"

def Read_PtNetRpt(inrpt):
    """Read net connections from PT report report_nets, return list of nets"""
    nets = {}
    with open(inrpt, 'r') as infile:
        linecount = 0
        for line in infile:
            linecount += 1
            if line.startswith('Connections'):
                try:
                    newnet
                except NameError:
                    pass
                else:
                    nets[newnet.name] = newnet
                index = line.split()
                newnet = Net()
                newnet.name = index[3].replace('\'', '').replace(':', '')
            index = line.split()
            if(len(index) > 1):
                if index[0] == 'total' and index[1] == 'capacitance:':
                    caps = re.findall(r'\d+\.\d+|\d+', line)
                    totalCap_min = float(caps[0])
                    totalCap_max = float(caps[1])
                    newnet.totalCap = (totalCap_min + totalCap_max)/2
                elif index[0] == 'wire' and index[1] == 'resistance':
                    res = re.findall(r'\d+\.\d+|\d+', line)
                    resistance_min = float(res[0])
                    resistance_max = float(res[1])
                    print(line, resistance_max, resistance_min)
                    newnet.resistance = (resistance_min + resistance_max)/2
                elif index[0] == 'Driver' and index[1] == 'Pins':
                    i = 1
                    while True:
                        newline = linecache.getline(str(inrpt), linecount+i)
                        index = newline.split()
                        if len(index) == 0:
                            break
                        elif '---' not in index[0]:
                            newnet.inpins.append(index[0])
                            if(index[2] == 'Port'):
                                newnet.isinPinPIPO.append(1)
                            else:
                                newnet.isinPinPIPO.append(0)
                            caps = re.findall(r'\d+\.\d+|\d+', newline)
                            newnet.inpin_caps.append([float(caps[0]), float(caps[1])])
                            
                        i += 1
                elif index[0] == 'Load' and index[1] == 'Pins':
                    i = 1
                    while True:
                        newline = linecache.getline(str(inrpt), linecount+i)
                        index = newline.split()
                        if len(index) == 0:
                            break
                        elif '---' not in index[0]:
                            newnet.outpins.append(index[0])
                            if(index[2] == 'Port'):
                                newnet.outpin_caps.append([float(index[3]), float(index[4])])
                                newnet.isoutPinPIPO.append(1)
                            else:
                                newnet.outpin_caps.append([float(index[4]), float(index[5])])
                                newnet.isoutPinPIPO.append(0)
                        i += 1
        nets[newnet.name] = newnet
    return nets