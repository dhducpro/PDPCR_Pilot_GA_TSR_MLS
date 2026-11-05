"""Offline conversion of the bundled Li-Lim LR2_2_6 data to PDP-CR.

python -m pdpcr.li_lim
Source pairing follows SINTEF's PDPTW documentation; conversion follows
section 5.1 of the supplied Thao paper. This is not the authors' converted file.
"""
from pathlib import Path
import hashlib
import json
import math
import random

from .core import Instance

ROOT = Path(__file__).resolve().parent.parent
INSTANCE_NAME = "lilim_LR2_2_6_n100_m10_req0"
SMALL_INSTANCE_NAME = "lilim_LR2_2_6_n6_m5_req0_first6"


def prefix_instance(parent, count=6, robots=5, name=SMALL_INSTANCE_NAME):
    """First complete tasks of a converted instance; never clip requirements."""
    if not 1 <= count <= parent.n:
        raise ValueError("Invalid number of selected requests")
    requirements = parent.requirements[1:count+1]
    if max(requirements) > robots:
        raise ValueError("A selected task requires more robots than the chosen fleet")
    indices = [0] + list(range(1, count+1)) + [parent.sink]
    return Instance(name, robots, [robots]+requirements+[robots],
                    [parent.processing[i] for i in indices],
                    [[parent.travel[i][j] for j in indices] for i in indices])


def convert_li_lim(path, robots=10, requirement_seed=0, expected_tasks=None):
    """Keep every pickup-delivery pair, ordered by the original pickup ID.

    c_i = Euclidean(pickup_i, delivery_i); t_ij = Euclidean(delivery_i, pickup_j).
    Source and sink share original depot coordinates. r_i ~ Uniform{1,...,5}
    using a private seeded RNG. Original TW, service, capacity and fleet ignored.
    """
    path = Path(path)
    raw = path.read_bytes()
    lines = [line.split() for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if not lines or len(lines[0]) != 3:
        raise ValueError("Expected the Li-Lim K/Q/S header")
    if not isinstance(robots, int) or robots < 5:
        raise ValueError("The paper's r_i in [1,5] requires at least five robots")
    nodes = {}
    for parts in lines[1:]:
        if len(parts) != 9:
            raise ValueError("Every Li-Lim node must have exactly nine fields")
        node = int(parts[0])
        if node in nodes:
            raise ValueError("Duplicate original node ID")
        values = [float(value) for value in parts[1:7]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite original node data")
        nodes[node] = dict(x=values[0], y=values[1], demand=values[2],
                           pickup=int(parts[7]), delivery=int(parts[8]))
    if 0 not in nodes:
        raise ValueError("Missing depot")
    tasks, used = [], set()
    for pickup_id, pickup in sorted(nodes.items()):
        if pickup_id == 0 or pickup["pickup"] != 0:
            continue
        delivery_id = pickup["delivery"]
        delivery = nodes.get(delivery_id)
        if not delivery_id or delivery is None or delivery["pickup"] != pickup_id or delivery["delivery"] != 0:
            raise ValueError("Broken pickup-delivery sibling relationship")
        if pickup_id in used or delivery_id in used or pickup_id == delivery_id:
            raise ValueError("Repeated node in task pairs")
        if pickup["demand"] <= 0 or delivery["demand"] != -pickup["demand"]:
            raise ValueError("Paired demands must have opposite signs and equal magnitude")
        used.update((pickup_id, delivery_id))
        tasks.append({"task": len(tasks) + 1, "pickup_id": pickup_id, "delivery_id": delivery_id,
                      "pickup": [pickup["x"], pickup["y"]],
                      "delivery": [delivery["x"], delivery["y"]]})
    if used != set(nodes) - {0}:
        raise ValueError("Some non-depot nodes are not part of a complete request")
    if expected_tasks is not None and len(tasks) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} complete tasks, found {len(tasks)}; no truncation")
    depot = [nodes[0]["x"], nodes[0]["y"]]
    pickups = [depot] + [task["pickup"] for task in tasks] + [depot]
    deliveries = [depot] + [task["delivery"] for task in tasks] + [depot]
    processing = [0.0] + [math.dist(task["pickup"], task["delivery"]) for task in tasks] + [0.0]
    travel = [[math.dist(deliveries[i], pickups[j]) for j in range(len(pickups))]
              for i in range(len(pickups))]
    rng = random.Random(requirement_seed)
    requirements = [robots] + [rng.randint(1, 5) for _ in tasks] + [robots]
    name = f"lilim_{path.stem}_n{len(tasks)}_m{robots}_req{requirement_seed}"
    instance = Instance(name, robots, requirements, processing, travel)
    metadata = {"instance": name, "source_file": path.name,
                "source_sha256": hashlib.sha256(raw).hexdigest(), "tasks": tasks,
                "task_count": len(tasks), "nondepot_nodes": len(used), "depot": depot,
                "robots": robots, "requirement_seed": requirement_seed,
                "requirements": requirements[1:-1],
                "processing_rule": "Euclidean distance pickup_i to delivery_i",
                "travel_rule": "Euclidean distance delivery_i to pickup_j; no rounding",
                "ignored_source_fields": ["time windows", "service time", "load capacity",
                                          "load demand", "original fleet size", "speed"],
                "author_instance_match": False,
                "note": "Independent conversion following paper section 5.1; authors' requirement seed and selected instance IDs are unavailable"}
    return instance, metadata


def main():
    folder = ROOT / "data" / "li_lim"
    source = folder / "LR2_2_6.pdp"
    provenance = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance["sha256"]:
        raise ValueError("Bundled raw source differs from its recorded hash")
    instance, metadata = convert_li_lim(source, expected_tasks=100)
    instance.save(ROOT / "instances" / f"{instance.name}.json")
    small = prefix_instance(instance)
    small.save(ROOT / "instances" / f"{small.name}.json")
    metadata["provenance"] = provenance
    metadata["reference_subset"] = {"instance": small.name, "robots": 5,
                                     "tasks": metadata["tasks"][:6],
                                     "selection": "first six complete requests; unchanged coordinates and requirements"}
    (folder / "conversion.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Converted {instance.n} full tasks / {metadata['nondepot_nodes']} nodes, "
          f"m={instance.robots}, requirement seed=0: {instance.name}")
    print(f"Small reference instance: {small.name}")


if __name__ == "__main__":
    main()
