from parsers import *
from work import work_var
import networkx as nx
import re
from typing import Union, Optional, List, Callable, Any
import logging
from utils import setup_logging
from utils.selected_cell import parse_cell_type
from pathlib import Path
import tqdm
from collections import defaultdict

def round_align(a: float, b: float):
    # align the decimal places of a and b
    def decimal_places(x):
        s = str(x)
        return len(s.split(".")[1]) if "." in s else 0
    
    dp = min(decimal_places(a), decimal_places(b))
    return round(a, dp), round(b, dp)

def format_data(val, sig_digits=6):
    if isinstance(val, float):
        return f"{val:.{sig_digits}g}"
    elif isinstance(val, list):
        return [format_data(v, sig_digits) for v in val]
    elif isinstance(val, tuple):
        return tuple(format_data(v, sig_digits) for v in val)
    elif isinstance(val, dict):
        return {k: format_data(v, sig_digits) for k, v in val.items()}
    else:
        return val
    
def _to_py_scalar(x: Any) -> Any:
    """Best-effort convert torch/numpy scalar to python scalar for safe boolean checks."""
    try:
        # torch / numpy scalar
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return x

class TimingGraph:
    """Timing Graph for test design"""
    """
    ############  following attributes are built by self.__init__  ###########
    self.design -> current design name
    self.cells -> a name dict for cells: cell refname (U222) -> cell class(name (ND2D0BWP16P90), ...)
    self.TNS -> [Setup, Hold] calculated by zero-out-degree nodes slack on timing graph
    self.WNS -> [Setup, Hold] calculated by zero-out-degree nodes slack on timing graph
    self.PT_WNS -> [Setup, Hold] read from PrimeTime global timing report
    self.PT_TNS -> [Setup, Hold] read from PrimeTime global timing report
    self.remove_flag -> bool flag to indicate if the unpropagated timing graph has been removed
    type_id / size_id are derived directly from the cell library name via parse_cell_type() in utils/selected_cell.py
    ##########################################################################
    
    ####  following attributes are built by self.remove_unpropagated_arcs  ###
    self.level_of -> dict[node] -> int level
    self.levels -> list[list[node]] nodes grouped by level
    self.level_nodes -> list[node] flattened by level, in topo order
    self.reversed_level_nodes -> list[node] flattened by level, in reversed topo order
    self.level_ptr -> list[int] prefix pointers of levels
    self.max_level -> int max level
    ##########################################################################
    
    ############  Core Attributes, Timing Graph of Current Design  ###########
    self.G -> DAG timing graphs
            G -> nodes: named with pin/port names
            features:
            "arrival": arrival time for pin and ports: [Max_Rise, Max_Fall, Min_Rise, Min_Fall]
            "ceff": effective capacitance calculated by PrimeTime for each driver pin: [Max_Rise_Ceff, Max_Fall_Ceff, Min_Rise_Ceff, Min_Fall_Ceff]
            "slack": slack for pin and ports: [Max_Rise, Max_Fall, Min_Rise, Min_Fall]
            "eco_slack": [target] slack for pin and ports after ECO: [Max_Rise, Max_Fall, Min_Rise, Min_Fall]
            "trans": transition time for pin and ports: [Max_Rise, Max_Fall, Min_Rise, Min_Fall]
            "leakage": PT total leakage for cell
            "is_port": if the node is a PI/PO: True/False
            "is_async_pin": if the pin is an asynchronous preset/clear pin
            "is_clock_network": if the pin is a combinational fanout of a clock
            "is_outpin": if the pin is a output pin of a cell
            "bbox": the bounding box of the pin
            "type": the type of the cell (BUFF, ND2, AOI23, etc.)
            "type_id": the type id of the cell i.e. 0 for BUFF, 1 for ND2, etc.
            "size_id": the size id of the cell i.e. 0 for BUFFD0BWP16P90, 1 for BUFFD1BWP16P90, etc.
            "size_id_eco": [target] the size id of the cell after ECO i.e. 0 for BUFFD0BWP16P90, 1 for BUFFD1BWP16P90, etc.
            "level": the level of the node
            "criticality": [target] if the cell is identified as critical by PrimeTimee
            
            G -> edges features:
            "name": cell/net name for different arcs
            "is_cell": if the edge is a cell arc
            "delay": the delay for the timing arc: [Max_Rise, Max_Fall, Min_Rise, Min_Fall]
            "sense_unate": the unate of the timing sense
            "when": when condition of the timing arc
            "sdf_cond": sdf condition of the timing arc
    ##########################################################################
            
    function:
    self.remove_unpropagated_arcs -> remove unnecessary timing arcs from the timing graph
    self.print_nodes -> output pin/port features for current design
    self.print_nodes_by_attr_condition -> output pin/port features for current design filtered by a condition on one node attribute
    self.print_edges -> output cell/net arc features for current design
    self.print_edges_by_attr_condition -> output cell/net arc features for current design filtered by a condition on one edge attribute
    self.print_level_info -> print the levelization information of the timing graph
    """
    
    def __init__(self, design: str, log: Optional[bool] = True) -> None:
        self.logger = logging.getLogger("TimingGraph")
        if log:
            setup_logging(self.logger, "INFO")
            self.logger.info(f"Initializing DAG timing graph of design {design} from EDA tools Rpt")
        else:
            logging.disable(logging.CRITICAL + 1)
        """Initialize DAG timing graph"""
        self.design = design
        # define cell and net Rpt Path
        CellRpt_Path =(
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_cell.rpt"
            )
        NetRpt_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_net.rpt"
            )
        # cells with name, type, inpins and outpins
        self.logger.info(f"Parsing PT Cell and Net Rpt")
        # U222 -> (type='ND2D0BWP16P90', name='ND2D0BWP16P90', inpins=[], outpins=[]) refname -> cell object
        self.cells = PtCellRpt_Parser.Read_PtCellRpt(CellRpt_Path.resolve())
        # nets with name inpins outpins resistance and capacitance
        nets = PtNetRpt_Parser.Read_PtNetRpt(NetRpt_Path.resolve())
        
        # define the PT timing arc Rpt Path
        TimingArcRpt_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_timing_arcs.rpt"
            )
        self.logger.info(f"Parsing PT Timing Arc Rpt")
        timing_arcs = PtTimingArc_Parser.Read_PtTimingArc(TimingArcRpt_Path.resolve())
        
        # Build Graph from Rpt items
        """ 
        Timing Graph G: multi edge DAG
        """
        self.G = nx.MultiDiGraph()
        """Noted, the delay is annotated in Max_Rise, Max_Fall, Min_Rise, Min_Fall order"""
        self.logger.info(f"Initializing Timing Graph with networkx MultiDiGraph")
        for timing_arc in tqdm.tqdm(
            timing_arcs, 
            desc="Building Timing Graph using Timing Arcs",
            colour="cyan",
            total=len(timing_arcs),
            ): 
            edge_data = {
                "is_cell": timing_arc.is_cell,
                "delay": timing_arc.delay,
                "sense_unate": timing_arc.sense_unate,
                "sdf_cond": timing_arc.sdf_cond,
                "when": timing_arc.when,
            }

            if timing_arc.is_cell:
                edge_data["name"] = timing_arc.from_pin.split("/")[0]

            self.G.add_edge(timing_arc.from_pin, timing_arc.to_pin, **edge_data)
        
        for net_name, net in tqdm.tqdm(
            nets.items(),
            desc="Annotating Timing Graph Net Edges",
            colour="cyan",
            total=len(nets),
        ):
            for inp in net.inpins:
                for outp in net.outpins:  
                    for key in self.G[inp][outp]:
                        self.G[inp][outp][key]["name"] = net_name
        
        """Reading data from PrimeTime"""
        # pin arrival time
        PinArrivalRpt_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_pin_arrival.rpt"
            )
        # port arrival time
        PortArrivalRpt_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_port_arrival.rpt"
            )
        # pin effective capacitance
        PinCapsRpt_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_pin_caps.rpt"
            )
        # slack
        PinSlackRpt_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_global_slack.rpt"
            )
        # pin transition time
        PinTrans_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_pin_transition.rpt"
            )
        # port transition time
        PortTrans_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_port_transition.rpt"
            )
        # cell leakage power
        CellLeakage_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_leakage.rpt"
            )
        # pin type
        PinType_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_pin_type.rpt"
            )
        # PT global timing report
        GlobalTimingRpt_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_glb_timing.rpt"
            )
        
        # PT ECO change list
        ECO_ChangeList_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_eco.tcl"
            )
        
        # PT ECO slack
        ECO_Slack_Path = (
                Path(work_var.pt_data_dir)
                / f"{design}"
                / f"{design}_global_slack_eco.rpt"
            )
        
        self.logger.info(f"Parsing PT Pin Attributes")
        # pins -> Max_Rise, Max_Fall, Min_Rise, Min_Fall arrival
        pins_arrival = PtPinArrival_Parser.Read_PtPinArrival(PinArrivalRpt_Path.resolve())
        # ports and Max_Rise, Max_Fall, Min_Rise, Min_Fall arrival
        ports_arrival = PtPortArrival_Parser.Read_PtPortArrival(PortArrivalRpt_Path.resolve())
        # pin and Max_Rise, Max_Fall, Min_Rise, Min_Fall effective capacitance
        pins_caps = PtPinCaps_Parser.Read_PtPinCaps(PinCapsRpt_Path.resolve())
        # pins -> Max_Rise, Max_Fall, Min_Rise, Min_Fall slack
        pins_ports_slack = PtPinSlack_Parser.Read_PtPinSlack(PinSlackRpt_Path.resolve())
        # pins -> Max_Rise, Max_Fall, Min_Rise, Min_Fall transition
        pins_trans = PtPinTrans_Parser.Read_PtPinTrans(PinTrans_Path.resolve())
        # ports -> Max_Rise, Max_Fall, Min_Rise, Min_Fall transition
        ports_trans = PtPortTrans_Parser.Read_PtPortTrans(PortTrans_Path.resolve())
        # cells -> Gate_Leakage, Intrinsic_Leakage,Totgal_Leakage
        cells_leakage = PtLeakage_Parser.Read_PtLeakage(CellLeakage_Path.resolve())
        # pins -> is_async_pin
        pin_types = PtPinType_Parser.Read_PtPinType(PinType_Path.resolve())
        # PT global timing report
        self.PT_WNS, self.PT_TNS = PtGlobalRpt_Parser.Read_GlobalRpt(GlobalTimingRpt_Path.resolve())
        # PT ECO change list (sizing only)
        size_solution, _ = PtChangeList_Parser.ReadChangeList(ECO_ChangeList_Path.resolve())
        # PT ECO slack
        ECO_pin_ports_slack = PtPinSlack_Parser.Read_PtPinSlack(ECO_Slack_Path.resolve())
                
        """Reading data from ICC2"""
        # pins properties
        Icc2PinRpt_Path = (
                Path(work_var.icc2_data_dir)
                / f"{design}"
                / "rpt"
                / f"{design}_pin.rpt"
            )
        
        # block properties
        Icc2BlockRpt_Path = (
                Path(work_var.icc2_data_dir)
                / f"{design}"
                / "rpt"
                / f"{design}_block.rpt"
            )

        self.logger.info(f"Parsing ICC2 Pin Attributes")

        # Icc2pins bbox -> name, llx, lly, urx, ury
        Icc2_pins = Icc2PinRpt_Parser.Read_Icc2PinRpt(Icc2PinRpt_Path)
        Icc2_block = Icc2BlockRpt_Parser.Read_Icc2BlockRpt(Icc2BlockRpt_Path)
        block_x = Icc2_block.urx - Icc2_block.llx
        block_y = Icc2_block.ury - Icc2_block.lly
        
        """ annotate timing/physical features on pins"""
        """ noted the timing parameters are annotated in Max_Rise, Max_Fall, Min_Rise, Min_Fall order"""
        self.TNS, self.WNS = [0.0, 0.0], [0.0, 0.0]
        # eco solution count
        eco_solution_count = 0
        
        for node in tqdm.tqdm(
            self.G.nodes, 
            desc="Annotating Timing Graph Nodes",
            colour="cyan",
            total=len(self.G.nodes),
            ): 
            # if node is a port
            if node in ports_trans.keys():
                port_trans = ports_trans[node]
                self.G.nodes[node].update({
                    "is_port": True,
                    "is_outpin": False,
                    # SetupRise -> Max_Rise, SetupFall -> Max_Fall, HoldRise -> Min_Rise, HoldFall -> Min_Fall
                    "trans": port_trans.trans,
                    "ceff": [None, None, None, None],
                    "leakage": None,
                    "type": "Port",
                    "size_id": -1,
                    "type_id": -1,
                    "size_id_eco": -1,
                })
                # arrival
                port_arrival = ports_arrival.get(node)
                self.G.nodes[node]["arrival"] = port_arrival.arrival if port_arrival else [None, None, None, None]
            
            # if the node is a pin
            else:
                self.G.nodes[node]["is_port"] = False
                # arrival time
                pin_arrival = pins_arrival[node]
                self.G.nodes[node]["arrival"] = pin_arrival.arrival
                # effective capacitance 
                pin_caps = pins_caps[node]
                self.G.nodes[node]["ceff"] = pin_caps.ceff
                # transition time
                pin_trans = pins_trans[node]
                self.G.nodes[node]["trans"] = pin_trans.trans
                # leakage power
                pin_leakage = cells_leakage[node.split("/")[0]]
                self.G.nodes[node]["leakage"] = pin_leakage.leakage[2]
                
                # celltype for sizing
                inst_name = node.split("/")[0]
                cell_inst = self.cells[inst_name]
                pin_name = node.split("/")[1]
                
                # attribute type_id and size_id for all pins
                type_id, size_id = parse_cell_type(cell_inst.type)
                self.G.nodes[node]["type_id"] = type_id
                self.G.nodes[node]["size_id"] = size_id
                
                # attribute size_id_eco for output pins
                if pin_name in cell_inst.outpins:
                    self.G.nodes[node]["is_outpin"] = True
                    # only attribute size_id_eco for output pins
                    if inst_name in size_solution.keys():
                        eco_solution_count += 1
                        _, size_id_eco = parse_cell_type(size_solution[inst_name])
                        self.G.nodes[node]["size_id_eco"] = size_id_eco
                    else:
                        self.G.nodes[node]["size_id_eco"] = self.G.nodes[node]["size_id"]
                else:
                    self.G.nodes[node]["is_outpin"] = False
                    self.G.nodes[node]["size_id_eco"] = -1
                    
                match = re.search(r"(.*?)D\d+BWP16P90", cell_inst.type)
                if match:
                    self.G.nodes[node]["type"] = match.group(1)
                else:
                    self.G.nodes[node]["type"] = "Port"
            
            # Icc2 physical attribute
            Icc2_pin = Icc2_pins.get(node)
            self.G.nodes[node]["bbox"] = [Icc2_pin.llx/block_x, Icc2_pin.lly/block_y, Icc2_pin.urx/block_x, Icc2_pin.ury/block_y]

            # slack
            pin_port_slack = pins_ports_slack.get(node)
            self.G.nodes[node]["slack"] = pin_port_slack.slack if pin_port_slack else [None, None, None, None]
            # ECO slack
            ECO_pin_port_slack = ECO_pin_ports_slack.get(node)
            self.G.nodes[node]["slack_eco"] = ECO_pin_port_slack.slack if ECO_pin_port_slack else [None, None, None, None]
            
            # estimate EP slack
            if self.G.out_degree(node) == 0:
                slack = [x if x is not None else 0 for x in self.G.nodes[node]["slack"]]
                self.TNS[0] += min(min(slack[0], slack[1]), 0)
                self.TNS[1] += min(min(slack[2], slack[3]), 0)
                self.WNS[0] = min(min(slack[0], slack[1]), self.WNS[0])
                self.WNS[1] = min(min(slack[2], slack[3]), self.WNS[1])
            
            # type
            type = pin_types[node]
            self.G.nodes[node]["is_async_pin"] = type[0]
            self.G.nodes[node]["is_clock_network"] = type[1]
        
        # eco solution count check, all solutions should be recorded
        if eco_solution_count != len(size_solution):
            self.logger.error(f"ECO solution count {eco_solution_count} does not match the number of cells in size_solution {len(self.size_solution)}")
            raise ValueError(f"ECO solution count {eco_solution_count} does not match the number of cells in size_solution {len(self.size_solution)}")
        
        self.logger.info(f"PrimeTime TNS: Setup {self.PT_TNS[0]}, Hold {self.PT_TNS[1]}, PrimeTime WNS: Setup {self.PT_WNS[0]}, Hold {self.PT_WNS[1]}")
        self.logger.info(f"Zero-Out-Degree TNS: Setup {self.TNS[0]}, Hold {self.TNS[1]}, Zero-Out-Degree WNS: Setup {self.WNS[0]}, Hold {self.WNS[1]}")
        self.remove_flag = False
        self.logger.info(f"Finished Initializing Timing Graph")
    
    def remove_unpropagated_arcs(self):
        """
        unnecessary timing arcs represents cell timing arcs which have timing sense unate other than negative_unate, positive_unate or rising_edge,
        or arcs which does not have delays
        These arcs belong to registers which are not considered for optimization
        """
        if self.remove_flag:
            self.logger.warning(f"Timing Graph has been removed, skipping")
            return
        
        unnecessary_arcs = []
        for u, v, k, data in tqdm.tqdm(
            self.G.edges(keys=True, data=True), 
            desc="Removing Unnecessary Arcs", 
            colour="cyan", 
            total=len(self.G.edges)
            ):
            if all(x == None for x in data["delay"]):
                unnecessary_arcs.append((u, v, k))
            elif (data["is_cell"]
                and data["sense_unate"] != "positive_unate"
                and data["sense_unate"] != "negative_unate"
                and data["sense_unate"] != "rising_edge"
                and data["sense_unate"] != "falling_edge"
            ):
                unnecessary_arcs.append((u, v, k))
            else:
                at1 = list(self.G.nodes[u]["arrival"])
                at1 = [x if x is not None else 0 for x in at1]
                at2 = list(self.G.nodes[v]["arrival"])
                at2 = [x if x is not None else 0 for x in at2]
                # if neither if at1[0,1] is smaller than at2[0,1], at2 is impossible to propagate through u
                at1_0, at2_0 = round_align(at1[0], at2[0])
                at1_1, at2_1 = round_align(at1[1], at2[1])
                if not (
                    at1_0 <= at2_0 and
                    at1_1 <= at2_1
                ) and not (
                    at1_0 <= at2_1 and
                    at1_1 <= at2_0
                ):
                    unnecessary_arcs.append((u, v, k))

        self.G.remove_edges_from(unnecessary_arcs)
        self.logger.warning(f"Unnecessary edges are removed from Timing Graph of {self.design}")
        
        # pins without slack are removed; it should be notice tha hold check will be affacted in "ac97"
        invalid_pins = [
            n for n, data in self.G.nodes(data=True)
            # if "slack" in data and all(s is None for s in data["slack"]) and data["is_clock_network"] or  all(a is None for a in data["arrival"])
            if "slack" in data and all(s is None for s in data["slack"])
            # if "arrival" in data and all(s is None for s in data["arrival"])
        ]
        self.G.remove_nodes_from(invalid_pins)
        self.logger.warning(f"Pins without slack are removed from Timing Graph of {self.design}")
        
        # isolated nodes are removed
        isolated_nodes = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolated_nodes)
        self.logger.warning(f"Isolated nodes are removed from Timing Graph of {self.design}")
        
        self.logger.info(f"Try to levelizing nodes of {self.design}")
        try:
            # topologically sort nodes, isolated nodes are not included
            self.level_nodes = list(nx.topological_sort(self.G))
            # levelization of nodes, and update self.levelized_nodes
            self._build_levels()
            # reversed levelized nodes for backward propagation in Lagrange_Solver
            self.reversed_level_nodes = self.level_nodes[::-1]
        except Exception as e:
            self.logger.error(f"Error occurred: {e}\n Timing Graph cannot be levelized.")
            self.logger.info(f"Trying to breakloop")
        self.remove_flag = True
        
    def get_setup_neg_ep(self):
        """
        get the EP nodes with negative setup slack
        """
        neg_ep = []
        for node, data in self.G.nodes(data=True):
            if self.G.out_degree(node) == 0:
                slack = [0 if x is None else x for x in data["slack"]]
                if min(slack[0], slack[1]) < 0:
                    neg_ep.append(node)
        return neg_ep
    
    def get_hold_neg_ep(self):
        """
        get the EP nodes with negative hold slack
        """
        neg_ep = []
        for node, data in self.G.nodes(data=True):
            if self.G.out_degree(node) == 0:
                slack = [0 if x is None else x for x in data["slack"]]
                if min(slack[2], slack[3]) < 0:
                    neg_ep.append(node)
        return neg_ep

    def _build_levels(self):
        """
        Build levelization for current DAG self.G (nx.MultiDiGraph).
        Outputs:
        - self.level_of: dict[node] -> int level
        - self.levels: list[list[node]] nodes grouped by level
        - self.level_nodes: list[node] flattened by level
        - self.level_ptr: list[int] prefix pointers of levels
        """
        
        G = self.G
        # 0) topo order
        topo = list(self.level_nodes)

        # 1) compute level[v] = 0 if no preds else 1 + max(level[pred])
        level_of = {}
        for v in topo:
            preds = list(G.predecessors(v))
            if not preds:
                level_of[v] = 0
                self.G.nodes[v]["level"] = 0
            else:
                # MultiDiGraph predecessor iteration ignores multi-edges, which is what we want for level.
                level_of[v] = 1 + max(level_of[p] for p in preds)
                self.G.nodes[v]["level"] = level_of[v]

        # 2) bucket by level
        buckets = defaultdict(list)
        max_level = 0
        for v in topo:
            lv = level_of[v]
            buckets[lv].append(v)
            if lv > max_level:
                max_level = lv

        # 3) make levels list
        levels = []
        for l in range(max_level + 1):
            nodes_l = buckets.get(l, [])
            levels.append(nodes_l)

        # 4) flatten -> level_nodes / level_ptr
        level_nodes = []
        level_ptr = [0]
        for nodes_l in levels:
            level_nodes.extend(nodes_l)
            level_ptr.append(len(level_nodes))

        # 5) save
        self.max_level = max_level
        self.level_of = level_of
        self.levels = levels
        self.level_nodes = level_nodes
        self.level_ptr = level_ptr

    def print_level_info(self):
        """
        print the level information of the timing graph
        """
        if not hasattr(self, "level_nodes"):
            self.logger.error(f"self.remove_unpropagated_arcs() have not been called before reporting level information")
            raise ValueError("Run self.remove_unpropagated_arcs() first before calling level information")
        
        for level, nodes in enumerate(self.levels):
            print(f"Level {level}: Total {len(nodes)} nodes")
        print(f"Max Level: {self.max_level}")
    
    def print_nodes(self, data_type: Optional[Union[str, List[str]]] = None, width: float = 30):
        """
        [DEBUG ONLY] print node name and node features in self.G
        Args:
            data_type: the list of type name of the data to print (example: ["bbox", "arrival"]), if None, print all data
            width: the width of the node name
        """
        if data_type and "level" in data_type and not hasattr(self, "level_nodes"):
            self.logger.error(f"self.remove_unpropagated_arcs() have not been called before reportining pin level")
            raise ValueError("Run self.remove_unpropagated_arcs() first before calling level information")
        
        print("------------------------------------------ Node Information ------------------------------------------")
        for node, data in self.G.nodes(data=True):
            node_str = str(node).ljust(width)
            if data_type:
                if isinstance(data_type, str):
                    assert data_type in data.keys()
                    print(f"Node: {node_str} | Data: {data[data_type]}")
                else:
                    print(f"Node: {node_str}")
                    for type in data_type:
                        assert type in data.keys()
                        print(f"  {type}: {format_data(data[type], 6)}")
            else:
                print(f"Node: {node_str} | Data: {data}")
        print("------------------------------------------------------------------------------------------------------\n")
    
    def print_nodes_by_attr_condition(
        self,
        print_attrs: Union[str, List[str]],
        cond_attr: str,
        cond: Callable[[Any], bool],
        *,
        width: float = 30,
        limit: Optional[int] = None,
        require_all_print_attrs: bool = False,
    ):
        """
        [DEBUG ONLY] Print nodes with filtered condition on one attribute, while printing other attributes.

        Args:
            print_attrs: attributes to print (e.g., ["arrival", "slack"])
            cond_attr: attribute used for filtering (e.g., "arrival")
            cond: predicate on cond_attr value (e.g., lambda x: x != 0)
            width: node name width
            limit: print at most N matched nodes
            require_all_print_attrs: if True, skip nodes missing any print_attrs; else print "N/A" for missing ones
        """
        if isinstance(print_attrs, str):
            print_attrs = [print_attrs]

        print("--------------------------- Filtered Node Information ---------------------------")
        print(f"Filter: {cond_attr} satisfies condition; Print: {print_attrs}")

        matched = 0
        printed = 0
        total = 0
        for node, data in self.G.nodes(data=True):
            total += 1
            if cond_attr not in data:
                continue
            # only use cond_attr to filter
            try:
                if not cond(data[cond_attr]):
                    continue
            except Exception as e:
                self.logger.warning(f"Condition failed on node {node}: {e}")
                continue

            # whether to print all print_attrs
            if require_all_print_attrs and any(a not in data for a in print_attrs):
                continue

            node_str = str(node).ljust(width)
            print(f"Node: {node_str}")

            for a in print_attrs:
                if a in data:
                    print(f"  {a}: {format_data(data[a], 6)}")
                else:
                    print(f"  {a}: None")
            printed += 1
            print()
            matched += 1
            if limit is not None and matched >= limit:
                break

        print(f"--------------------------- Matched Nodes: {matched} | Printed Nodes: {printed} | Total Nodes: {total} ---------------------------\n")
    
    def print_edges(self, data_type: Optional[Union[str, List[str]]] = None, width: float = 30):
        """
        [DEBUG ONLY] print edge name and edge features in self.G
        Args:
            data_type: the list of type name of the data to print (example: ["delay", "sense_unate"]), if None, print all data
            width: the width of the edge name
        """
        print("------------------------------------------------------------------------------------ Edge Information ------------------------------------------------------------------------------------")
        for u, v, key, data in self.G.edges(keys=True, data=True):
            u_str = str(u).ljust(width)
            v_str = str(v).ljust(width)
            if data_type:
                if isinstance(data_type, str):
                    assert data_type in data.keys()
                    print(f"Edge: {u_str} -> {v_str} :edge{key} | Data: {data[data_type]}")
                else:
                    for type in data_type:
                        assert type in data.keys()
                    data_str = ", ".join(f"{type}: {format_data(data[type], 6)}" for type in data_type)
                    print(f"Edge: {u_str} -> {v_str} :edge{key} | {data_str}")
            else:
                print(f"Edge: {u_str} -> {v_str} :edge{key} | Data: {data}")
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

    def print_edges_by_attr_condition(
        self,
        print_attrs: Optional[Union[str, List[str]]] = None,
        cond_attr: Optional[str] = None,
        cond: Optional[Callable[[Any], bool]] = None,
        *,
        width: float = 30,
        limit: Optional[int] = None,
        require_all_print_attrs: bool = False,
    ):
        """
        [DEBUG ONLY] Print edges filtered by a condition on one edge attribute.

        Args:
            print_attrs:
                edge attributes to print (e.g., ["delay", "sense", "delay_grad"]).
                If None, print all edge data dict.

            cond_attr:
                edge attribute used for filtering (e.g., "delay_grad").

            cond:
                predicate function on cond_attr value, e.g.:
                    lambda x: x != 0
                    lambda x: abs(x) > 1e-6

            width:
                width for u/v formatting

            limit:
                print at most N matched edges; still counts total matched.

            require_all_print_attrs:
                if True, skip edges missing any print_attrs;
                if False, print "N/A" for missing ones.
        """
        if isinstance(print_attrs, str):
            print_attrs = [print_attrs]

        if (cond_attr is None) != (cond is None):
            raise ValueError("cond_attr and cond must be both provided or both None.")

        print("-------------------------------------------------------------- Filtered Edge Information --------------------------------------------------------------")
        if cond_attr is not None:
            print(f"Filter: {cond_attr} satisfies condition; Print: {print_attrs if print_attrs is not None else 'ALL'}")
        else:
            print(f"No filter; Print: {print_attrs if print_attrs is not None else 'ALL'}")

        total_matched = 0
        printed = 0
        total = 0

        for u, v, key, data in self.G.edges(keys=True, data=True):
            total += 1
            # 1) filtering
            if cond_attr is not None:
                if cond_attr not in data:
                    continue
                try:
                    val = _to_py_scalar(data[cond_attr])
                    ok = bool(cond(val))
                except Exception as e:
                    self.logger.warning(f"Condition failed on edge ({u}->{v}:edge{key}): {e}")
                    continue
                if not ok:
                    continue

            total_matched += 1
            # 2) printing limit (do not break; keep counting)
            if limit is not None and printed >= limit:
                continue

            u_str = str(u).ljust(width)
            v_str = str(v).ljust(width)

            # 3) print selected attrs or all
            if print_attrs is None:
                print(f"Edge: {u_str} -> {v_str} :edge{key} | Data: {data}")
            else:
                if require_all_print_attrs and any(a not in data for a in print_attrs):
                    continue

                parts = []
                for a in print_attrs:
                    if a in data:
                        parts.append(f"{a}: {format_data(data[a], 6)}")
                    else:
                        parts.append(f"{a}: N/A")
                data_str = ", ".join(parts)
                print(f"Edge: {u_str} -> {v_str} :edge{key} | {data_str}")

            printed += 1

        print(f"-------------------------------------------------------------- Total matched: {total_matched} | Printed: {printed} | Total: {total} --------------------------------------------------------------\n")