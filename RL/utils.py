from langchain.messages import HumanMessage


def trajectory_to_judge_text(trajectory):
    tool_calls = []
    final_answer = None

    for item in trajectory:
        if item["role"] == "tool":
            tool_calls.append(item.get("tool_name", "unknown_tool"))
        elif item["role"] == "ai":
            final_answer = item["content"]

    tool_part = (
        f"Tools used: {', '.join(tool_calls)}" if tool_calls else "Tools used: None"
    )

    answer_part = f"Final answer: {final_answer}"

    return f"{tool_part}\n{answer_part}"


import random


async def judge_pair(judge, prompt, traj_a, traj_b):
    # ===== 随机映射，去 position bias =====
    if random.random() < 0.5:
        label_a, label_b = "A", "B"
        text_a, text_b = traj_a, traj_b
        reverse = False
    else:
        label_a, label_b = "B", "A"
        text_a, text_b = traj_b, traj_a
        reverse = True

    content = f"""
You are an impartial judge.

User prompt:
{prompt}

{label_a}:
{trajectory_to_judge_text(text_a)}

{label_b}:
{trajectory_to_judge_text(text_b)}

Evaluation criteria:
- Correctness of the final answer
- Appropriate use of tools (if needed)
- No hallucination or unnecessary steps

Which is better? Answer ONLY "{label_a}" or "{label_b}".
"""

    resp = await judge.ainvoke([HumanMessage(content=content)])

    answer = resp.content.strip()

    winner = label_a if label_a in answer else label_b

    # ===== 还原到原始 trajectory =====
    if reverse:
        return "A" if winner == "B" else "B"
    return winner


async def select_best_and_worst(judge, prompt, trajectories):
    assert len(trajectories) >= 2

    # ===== best =====
    best = trajectories[0]
    for t in trajectories[1:]:
        winner = await judge_pair(judge, prompt, best, t)
        if winner == "B":
            best = t

    # ===== worst =====
    worst = trajectories[0]
    for t in trajectories[1:]:
        winner = await judge_pair(judge, prompt, worst, t)
        if winner == "A":
            worst = t

    return best, worst
