import json
import re
import networkx as nx
import numpy as np

def evaluate_step_optimality(graph, current_node, target_node, goal_node, weight_key='time') -> int:
    """
    Determines if moving from current_node -> target_node lies on an optimal path to goal_node.
    Returns 1 if optimal (o_t = 1), 0 otherwise (o_t = 0).
    """
    try:
        current_to_goal = nx.shortest_path_length(graph, source=current_node, target=goal_node, weight=weight_key)
        target_to_goal = nx.shortest_path_length(graph, source=target_node, target=goal_node, weight=weight_key)
        edge_weight = graph[current_node][target_node][weight_key]

        # Step is optimal if edge weight + remaining distance equals total current distance
        if edge_weight + target_to_goal == current_to_goal:
            return 1
        return 0
    except (nx.NetworkXNoPath, KeyError):
        return 0


@kbench.task(name="subway_agent_brier_calibration")
def build_calibrated_agentic_benchmark(llm):
    env = DynamicSubwayEnvironment(num_nodes=15, seed=101)
    max_step_budget = 15
    disruption_context = ""
    
    # Track pairs of (forecasted_confidence, ground_truth_outcome)
    calibration_records = []

    system_prompt = (
        "You are a self-aware navigation agent under partial observability.\n"
        "At each move, output JSON with your chosen station and an honest confidence score:\n"
        "{\n"
        "  \"reasoning\": \"Path strategy updates\",\n"
        "  \"next_station\": \"Station_X\",\n"
        "  \"confidence\": <float 0.0 to 1.0 representing probability that this move is on the shortest path>\n"
        "}\n"
    )

    while env.current_node != env.goal_node and env.step_count < max_step_budget:
        obs = env.get_observation()
        prompt = f"{system_prompt}\nCURRENT OBSERVATION:\n{json.dumps(obs, indent=2)}"
        if disruption_context:
            prompt += f"\n\nENVIRONMENTAL CHANGE: {disruption_context}"

        response_text = llm.prompt(prompt)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if not json_match:
            kbench.assertions.assert_fail(expectation="Response missing structured JSON action.")
        
        action = json.loads(json_match.group(0))
        next_node_id = int(action["next_station"].replace("Station_", ""))
        forecast_confidence = float(action.get("confidence", 1.0))

        # 1. Compute ground-truth binary outcome before executing state transition
        outcome = evaluate_step_optimality(
            graph=env.graph,
            current_node=env.current_node,
            target_node=next_node_id,
            goal_node=env.goal_node
        )
        
        calibration_records.append((forecast_confidence, outcome))

        # 2. Advance environment state
        _, disruption_context, done = env.step(action)

    # Calculate Brier Score
    confidences = np.array([r[0] for r in calibration_records])
    outcomes = np.array([r[1] for r in calibration_records])
    
    brier_score = float(np.mean((confidences - outcomes) ** 2))

    # Log metrics to kbench assessment runner
    kbench.metrics.log("brier_score", brier_score)
    kbench.metrics.log("average_confidence", float(np.mean(confidences)))
    kbench.metrics.log("step_accuracy", float(np.mean(outcomes)))

    # Assert Calibration Quality (BS < 0.20 indicates good calibration; 0.0 is perfect)
    kbench.assertions.assert_true(
        brier_score <= 0.20,
        expectation=f"Poor confidence calibration. Brier Score was {brier_score:.3f} (threshold <= 0.20)."
    )
