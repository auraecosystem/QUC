import json
import re
import networkx as nx
import numpy as np

class DynamicQuantumQPUEnvironment:
    """
    POMDP Environment simulating physical qubit routing on a noisy QPU topology.
    """
    def __init__(self, num_qubits=12, seed=42):
        np.random.seed(seed)
        # Generate physical hardware coupling map (Grid Topology)
        self.hardware_graph = nx.grid_2d_graph(3, 4)
        self.hardware_graph = nx.convert_node_labels_to_integers(self.hardware_graph)
        
        # Edge attributes: Gate Error Rate (Infidelity) and Execution Duration (ns)
        for u, v in self.hardware_graph.edges():
            self.hardware_graph[u][v]['error_rate'] = float(np.random.uniform(0.005, 0.03))
            self.hardware_graph[u][v]['duration_ns'] = int(np.random.randint(200, 500))

        self.start_qubit = 0
        self.target_qubit = num_qubits - 1
        self.current_qubit = self.start_qubit
        self.step_count = 0
        self.accumulated_fidelity = 1.0
        self.accumulated_time_ns = 0
        self.disruption_event = False

    def get_observation((self) -> dict:
        """Exposes local coupling map and local calibration data."""
        adjacent_couplers = []
        for nbr, attrs in self.hardware_graph[self.current_qubit].items():
            adjacent_couplers.append({
                "connected_qubit": f"Q_{nbr}",
                "gate_error_rate": round(attrs['error_rate'], 4),
                "gate_duration_ns": attrs['duration_ns']
            })
        return {
            "current_physical_qubit": f"Q_{self.current_qubit}",
            "target_entanglement_qubit": f"Q_{self.target_qubit}",
            "routing_steps": self.step_count,
            "accumulated_circuit_fidelity": round(self.accumulated_fidelity, 4),
            "total_execution_time_ns": self.accumulated_time_ns,
            "available_couplers": adjacent_couplers
        }

    def step(self, action_dict: dict):
        next_q = int(action_dict["next_qubit"].replace("Q_", ""))
        
        if not self.hardware_graph.has_edge(self.current_qubit, next_q):
            raise ValueError(f"Invalid Swap: Coupler between Q_{self.current_qubit} and Q_{next_q} does not exist.")

        # Dynamic Noise Drift (Injected at Step 2)
        disruption_msg = None
        if self.step_count == 2 and not self.disruption_event:
            self.disruption_event = True
            # Simulate dynamic noise spike due to cross-talk on an adjacent qubit
            out_edges = list(self.hardware_graph.edges(next_q))
            if out_edges:
                u, v = out_edges[0]
                self.hardware_graph[u][v]['error_rate'] = min(0.35, self.hardware_graph[u][v]['error_rate'] * 10)
                disruption_msg = f"CALIBRATION DRIFT ALERT: Severe error rate spike on coupler Q_{u} <-> Q_{v} (Error = {self.hardware_graph[u][v]['error_rate']:.3f})."

        edge_attrs = self.hardware_graph[self.current_qubit][next_q]
        self.accumulated_fidelity *= (1.0 - edge_attrs['error_rate'])
        self.accumulated_time_ns += edge_attrs['duration_ns']
        self.current_qubit = next_q
        self.step_count += 1

        done = (self.current_qubit == self.target_qubit)
        return self.get_observation(), disruption_msg, done


def evaluate_quantum_step_optimality(hardware_graph, current_q, target_q, goal_q) -> int:
    """
    Determines if routing through target_q yields optimal path fidelity to goal_q.
    Using log-transformed error weights: -log(1 - error_rate).
    """
    for u, v in hardware_graph.edges():
        err = hardware_graph[u][v]['error_rate']
        hardware_graph[u][v]['cost'] = -np.log(max(1e-6, 1.0 - err))

    try:
        curr_dist = nx.shortest_path_length(hardware_graph, source=current_q, target=goal_q, weight='cost')
        next_dist = nx.shortest_path_length(hardware_graph, source=target_q, target=goal_q, weight='cost')
        edge_cost = hardware_graph[current_q][target_q]['cost']

        if np.isclose(edge_cost + next_dist, curr_dist):
            return 1
        return 0
    except (nx.NetworkXNoPath, KeyError):
        return 0


@kbench.task(name="quantum_qubit_routing_calibration")
def build_quantum_agentic_benchmark(llm):
    env = DynamicQuantumQPUEnvironment(num_qubits=12, seed=2026)
    max_step_budget = 10
    disruption_context = ""
    calibration_records = []

    system_prompt = (
        "You are an AI Quantum Compiler routing multi-qubit entangling gates across physical hardware.\n"
        "Optimize for Maximum Final Circuit Fidelity by avoiding noisy couplers.\n"
        "Output strictly in JSON format:\n"
        "{\n"
        "  \"reasoning\": \"Analysis of hardware noise map and fidelity decay\",\n"
        "  \"next_qubit\": \"Q_X\",\n"
        "  \"confidence\": <float 0.0 to 1.0 representing predicted probability that this move preserves maximum state fidelity>\n"
        "}\n"
    )

    while env.current_qubit != env.target_qubit and env.step_count < max_step_budget:
        obs = env.get_observation()
        prompt = f"{system_prompt}\nHARDWARE TELEMETRY:\n{json.dumps(obs, indent=2)}"
        if disruption_context:
            prompt += f"\n\nHARDWARE DRIFT EVENT: {disruption_context}"

        response_text = llm.prompt(prompt)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if not json_match:
            kbench.assertions.assert_fail(expectation="Output must contain valid JSON action block.")
        
        action = json.loads(json_match.group(0))
        next_q_id = int(action["next_qubit"].replace("Q_", ""))
        forecast_confidence = float(action.get("confidence", 1.0))

        # Evaluate Step Optimality & Track Calibration
        outcome = evaluate_quantum_step_optimality(
            hardware_graph=env.hardware_graph,
            current_q=env.current_qubit,
            target_q=next_q_id,
            goal_q=env.target_qubit
        )
        calibration_records.append((forecast_confidence, outcome))

        _, disruption_context, done = env.step(action)

    # Compute Quantum Metacognition (Brier Score)
    confidences = np.array([r[0] for r in calibration_records])
    outcomes = np.array([r[1] for r in calibration_records])
    brier_score = float(np.mean((confidences - outcomes) ** 2))

    kbench.metrics.log("quantum_brier_score", brier_score)
    kbench.metrics.log("final_circuit_fidelity", env.accumulated_fidelity)

    # Assert Goal Reachability & Metacognitive Calibration
    kbench.assertions.assert_true(
        env.current_qubit == env.target_qubit,
        expectation="Agent failed to route qubit to destination within step budget."
    )
    kbench.assertions.assert_true(
        brier_score <= 0.20,
        expectation=f"Miscalibrated quantum confidence. Brier Score was {brier_score:.3f} (threshold <= 0.20)."
    )
