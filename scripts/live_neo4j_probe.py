from __future__ import annotations

import argparse
import os
import time

from neo4j import GraphDatabase

from src.http_server import build_multi_departure_response, load_network


def find_stop_candidates() -> list[tuple[str, str]]:
    query = (
        "MATCH (s:Stop) "
        "WHERE toLower(coalesce(s.name, '')) CONTAINS 'lausanne' "
        "   OR toLower(coalesce(s.name, '')) CONTAINS 'geneve' "
        "   OR toLower(coalesce(s.name, '')) CONTAINS 'genève' "
        "   OR toLower(coalesce(s.name, '')) CONTAINS 'geneva' "
        "RETURN s.stop_id AS stop_id, coalesce(s.name, '') AS stop_name "
        "LIMIT 100"
    )
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session() as session:
            return [(record["stop_id"], record["stop_name"]) for record in session.run(query)]
    finally:
        driver.close()


def select_ids(stop_candidates: list[tuple[str, str]]) -> tuple[str | None, str | None]:
    for stop_id, stop_name in stop_candidates:
        if stop_name.strip().lower() == "lausanne":
            lausanne = stop_id
            break
    else:
        lausanne = None

    for stop_id, stop_name in stop_candidates:
        lowered = stop_name.strip().lower()
        if lowered == "genève" or lowered == "geneve":
            geneva = stop_id
            break
    else:
        geneva = None

    if lausanne is not None and geneva is not None:
        return lausanne, geneva

    lausanne = None
    geneva = None
    for stop_id, stop_name in stop_candidates:
        text = stop_name.lower()
        if lausanne is None and "laus" in text:
            lausanne = stop_id
        if geneva is None and ("genev" in text or "geneve" in text or "genève" in text):
            geneva = stop_id
    return lausanne, geneva


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--start-stop-id", default=None)
    parser.add_argument("--end-stop-id", default=None)
    args = parser.parse_args()

    candidates = find_stop_candidates()
    print(f"candidate_count={len(candidates)}")
    print(candidates[:20])

    if args.list_only:
        return 0

    start_id = args.start_stop_id
    end_id = args.end_stop_id
    if start_id is None or end_id is None:
        start_id, end_id = select_ids(candidates)
    if start_id is None or end_id is None:
        print("could_not_auto_select_lausanne_geneva")
        return 2

    print(f"selected_start={start_id} selected_end={end_id}")

    t0 = time.perf_counter()
    network = load_network()
    load_s = time.perf_counter() - t0
    print(
        f"network_loaded stops={len(network.stop_ids)} trips={len(network.trip_ids)} routes={len(network.route_trip_offsets)-1} load_s={load_s:.2f}"
    )

    if start_id not in network.stop_id_index or end_id not in network.stop_id_index:
        print("selected_ids_not_in_loaded_network")
        return 3

    start_idx = network.stop_id_index[start_id]
    end_idx = network.stop_id_index[end_id]

    solve_t0 = time.perf_counter()
    response = build_multi_departure_response(
        network,
        "raptor",
        [start_idx],
        end_idx,
        8 * 3600,
        offset_minutes=(0,),
    )
    solve_s = time.perf_counter() - solve_t0

    segments = response.get("segments") or []
    print(
        f"solve_s={solve_s:.2f} segments={len(segments)} fallback_used={response.get('fallback_used')} resolver_algorithm={response.get('resolver_algorithm')}"
    )
    if segments:
        print(f"arrival={segments[-1].get('arrival_time')} transfers={response.get('transfers')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
