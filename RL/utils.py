def trajectory_to_judge_text(trajectory):
    tool_calls = []
    final_answer = ""

    for item in trajectory:
        if item["role"] == "tool":
            tool_calls.append(item.get("tool_name", "unknown_tool"))
        if item["role"] == "ai":
            final_answer = item["content"]

    return (
        f"- Tool used: {', '.join(tool_calls) if tool_calls else 'None'}\n"
        f"- Final answer: {final_answer}"
    )


async def judge_pair(judge, prompt, traj_a, traj_b):
    content = f"""
User prompt:
{prompt}

Trajectory A:
{trajectory_to_judge_text(traj_a)}

Trajectory B:
{trajectory_to_judge_text(traj_b)}

Which trajectory is better? Answer ONLY "A" or "B".
"""

    resp = await judge.ainvoke({"messages": [{"role": "user", "content": content}]})

    answer = resp.content.strip()
    return "A" if "A" in answer else "B"


async def select_best_and_worst(prompt, trajectories):
    best = trajectories[0]

    for t in trajectories[1:]:
        winner = await judge_pair(prompt, best, t)
        if winner == "B":
            best = t

    worst = trajectories[0]
    for t in trajectories[1:]:
        loser = await judge_pair(prompt, worst, t)
        if loser == "A":
            worst = t

    return best, worst
