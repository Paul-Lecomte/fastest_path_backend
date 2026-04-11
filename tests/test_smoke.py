# This file contains smoke tests for the RAPTOR, Dijkstra, and A* algorithms on a mock transportation network.
import asyncio
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from src.loader import (
    TransitNetwork,
    STOPS_DTYPE,
    ROUTES_DTYPE,
    STOP_TIMES_DTYPE,
    _build_adjacency,
    _build_routes,
    _build_transfers,
    _build_trip_cost_factors,
    build_mock_network,
)
from src.solver import build_path, build_path_dijkstra, run_dijkstra_fast, run_raptor, run_astar_fast
from src.http_server import (
    build_multi_departure_response,
    _departure_to_seconds,
    _build_segment_payloads,
    _select_starts_from_origin,
    _select_ends_from_destination,
    _resolve_stop_query_to_index,
)
from src.server import _find_fastest_segments_parallel, _get_start_stop_ids


def _build_linear_transfer_network(stop_count: int = 12) -> TransitNetwork:
    stop_ids = [f"S{i}" for i in range(stop_count)]
    trip_ids = [f"T{i}" for i in range(stop_count - 1)]

    stops_array = np.zeros(len(stop_ids), dtype=STOPS_DTYPE)
    for i in range(len(stop_ids)):
        stops_array[i] = (i, i)

    rows = []
    for trip_id in range(stop_count - 1):
        departure = 1000 + trip_id * 120
        arrival = departure + 60
        rows.append((trip_id, trip_id, departure, 1))
        rows.append((trip_id + 1, trip_id, arrival, 2))

    stop_times_array = np.array(rows, dtype=STOP_TIMES_DTYPE)

    routes_array = np.zeros(len(trip_ids), dtype=ROUTES_DTYPE)
    for i in range(len(trip_ids)):
        routes_array[i] = (i, i)

    trip_offsets = np.zeros(len(trip_ids) + 1, dtype=np.int64)
    for trip_id in range(len(trip_ids)):
        trip_offsets[trip_id] = trip_id * 2
    trip_offsets[len(trip_ids)] = len(rows)

    adj_offsets, adj_neighbors, adj_weights, adj_trip_ids = _build_adjacency(
        stop_times_array,
        trip_offsets,
        len(stop_ids),
    )

    (
        route_stop_offsets,
        route_stops,
        route_trip_offsets,
        route_trips,
        route_board_offsets,
        route_board_times,
        route_board_monotonic,
        stop_route_offsets,
        stop_routes,
    ) = _build_routes(stop_times_array, trip_offsets, len(stop_ids))

    stop_id_index = {value: idx for idx, value in enumerate(stop_ids)}
    trip_id_index = {value: idx for idx, value in enumerate(trip_ids)}
    stop_lats = np.full(len(stop_ids), np.nan, dtype=np.float64)
    stop_lons = np.full(len(stop_ids), np.nan, dtype=np.float64)

    transfer_offsets, transfer_neighbors, transfer_weights = _build_transfers(
        stop_ids,
        stop_lats,
        stop_lons,
    )

    return TransitNetwork(
        stops=stops_array,
        stop_times=stop_times_array,
        routes=routes_array,
        stop_id_index=stop_id_index,
        trip_id_index=trip_id_index,
        stop_ids=stop_ids,
        stop_lats=stop_lats,
        stop_lons=stop_lons,
        trip_ids=trip_ids,
        trip_offsets=trip_offsets,
        adj_offsets=adj_offsets,
        adj_neighbors=adj_neighbors,
        adj_weights=adj_weights,
        adj_trip_ids=adj_trip_ids,
        route_stop_offsets=route_stop_offsets,
        route_stops=route_stops,
        route_trip_offsets=route_trip_offsets,
        route_trips=route_trips,
        route_board_offsets=route_board_offsets,
        route_board_times=route_board_times,
        route_board_monotonic=route_board_monotonic,
        stop_route_offsets=stop_route_offsets,
        stop_routes=stop_routes,
        transfer_offsets=transfer_offsets,
        transfer_neighbors=transfer_neighbors,
        transfer_weights=transfer_weights,
    )


def _build_destination_trap_network() -> TransitNetwork:
    stop_ids = ["A", "B", "C", "D"]
    trip_ids = ["T0"]

    stops_array = np.zeros(len(stop_ids), dtype=STOPS_DTYPE)
    for i in range(len(stop_ids)):
        stops_array[i] = (i, i)

    stop_times_array = np.array(
        [
            (0, 0, 1000, 1),
            (1, 0, 1050, 2),
            (2, 0, 1100, 3),
        ],
        dtype=STOP_TIMES_DTYPE,
    )

    routes_array = np.zeros(len(trip_ids), dtype=ROUTES_DTYPE)
    routes_array[0] = (0, 0)

    trip_offsets = np.array([0, 3], dtype=np.int64)

    adj_offsets, adj_neighbors, adj_weights, adj_trip_ids = _build_adjacency(
        stop_times_array,
        trip_offsets,
        len(stop_ids),
    )

    (
        route_stop_offsets,
        route_stops,
        route_trip_offsets,
        route_trips,
        route_board_offsets,
        route_board_times,
        route_board_monotonic,
        stop_route_offsets,
        stop_routes,
    ) = _build_routes(stop_times_array, trip_offsets, len(stop_ids))

    stop_id_index = {value: idx for idx, value in enumerate(stop_ids)}
    trip_id_index = {value: idx for idx, value in enumerate(trip_ids)}
    stop_lats = np.array([46.5, 46.51, 46.52, 46.5260], dtype=np.float64)
    stop_lons = np.array([6.6, 6.61, 6.62, 6.6260], dtype=np.float64)
    transfer_offsets, transfer_neighbors, transfer_weights = _build_transfers(
        stop_ids,
        stop_lats,
        stop_lons,
    )

    return TransitNetwork(
        stops=stops_array,
        stop_times=stop_times_array,
        routes=routes_array,
        stop_id_index=stop_id_index,
        trip_id_index=trip_id_index,
        stop_ids=stop_ids,
        stop_lats=stop_lats,
        stop_lons=stop_lons,
        trip_ids=trip_ids,
        trip_offsets=trip_offsets,
        adj_offsets=adj_offsets,
        adj_neighbors=adj_neighbors,
        adj_weights=adj_weights,
        adj_trip_ids=adj_trip_ids,
        route_stop_offsets=route_stop_offsets,
        route_stops=route_stops,
        route_trip_offsets=route_trip_offsets,
        route_trips=route_trips,
        route_board_offsets=route_board_offsets,
        route_board_times=route_board_times,
        route_board_monotonic=route_board_monotonic,
        stop_route_offsets=stop_route_offsets,
        stop_routes=stop_routes,
        transfer_offsets=transfer_offsets,
        transfer_neighbors=transfer_neighbors,
        transfer_weights=transfer_weights,
    )


def _build_mode_preference_network() -> TransitNetwork:
    stop_ids = ["A", "C"]
    trip_ids = ["BUS_DIRECT", "TRAIN_DIRECT"]

    stops_array = np.zeros(len(stop_ids), dtype=STOPS_DTYPE)
    for i in range(len(stop_ids)):
        stops_array[i] = (i, i)

    stop_times_array = np.array(
        [
            (0, 0, 1000, 1),
            (1, 0, 1200, 2),
            (0, 1, 1030, 1),
            (1, 1, 1250, 2),
        ],
        dtype=STOP_TIMES_DTYPE,
    )

    routes_array = np.zeros(len(trip_ids), dtype=ROUTES_DTYPE)
    routes_array[0] = (0, 0)
    routes_array[1] = (1, 1)

    trip_offsets = np.array([0, 2, 4], dtype=np.int64)

    adj_offsets, adj_neighbors, adj_weights, adj_trip_ids = _build_adjacency(
        stop_times_array,
        trip_offsets,
        len(stop_ids),
    )

    (
        route_stop_offsets,
        route_stops,
        route_trip_offsets,
        route_trips,
        route_board_offsets,
        route_board_times,
        route_board_monotonic,
        stop_route_offsets,
        stop_routes,
    ) = _build_routes(stop_times_array, trip_offsets, len(stop_ids))

    stop_id_index = {value: idx for idx, value in enumerate(stop_ids)}
    trip_id_index = {value: idx for idx, value in enumerate(trip_ids)}
    stop_lats = np.array([46.5, 46.52], dtype=np.float64)
    stop_lons = np.array([6.6, 6.62], dtype=np.float64)
    transfer_offsets, transfer_neighbors, transfer_weights = _build_transfers(stop_ids, stop_lats, stop_lons)
    trip_route_types = np.array([3, 2], dtype=np.int16)
    trip_cost_factors = _build_trip_cost_factors(trip_route_types)

    return TransitNetwork(
        stops=stops_array,
        stop_times=stop_times_array,
        routes=routes_array,
        stop_id_index=stop_id_index,
        trip_id_index=trip_id_index,
        stop_ids=stop_ids,
        stop_lats=stop_lats,
        stop_lons=stop_lons,
        trip_ids=trip_ids,
        trip_offsets=trip_offsets,
        adj_offsets=adj_offsets,
        adj_neighbors=adj_neighbors,
        adj_weights=adj_weights,
        adj_trip_ids=adj_trip_ids,
        route_stop_offsets=route_stop_offsets,
        route_stops=route_stops,
        route_trip_offsets=route_trip_offsets,
        route_trips=route_trips,
        route_board_offsets=route_board_offsets,
        route_board_times=route_board_times,
        route_board_monotonic=route_board_monotonic,
        stop_route_offsets=stop_route_offsets,
        stop_routes=stop_routes,
        trip_route_types=trip_route_types,
        trip_cost_factors=trip_cost_factors,
        transfer_offsets=transfer_offsets,
        transfer_neighbors=transfer_neighbors,
        transfer_weights=transfer_weights,
    )


def _build_short_walk_vs_shuttle_network() -> TransitNetwork:
    stop_ids = ["A", "B", "C"]
    trip_ids = ["TRAIN_AB", "BUS_BC"]

    stops_array = np.zeros(len(stop_ids), dtype=STOPS_DTYPE)
    for i in range(len(stop_ids)):
        stops_array[i] = (i, i)

    stop_times_array = np.array(
        [
            (0, 0, 900, 1),
            (1, 0, 1000, 2),
            (1, 1, 1100, 1),
            (2, 1, 1200, 2),
        ],
        dtype=STOP_TIMES_DTYPE,
    )

    routes_array = np.zeros(len(trip_ids), dtype=ROUTES_DTYPE)
    routes_array[0] = (0, 0)
    routes_array[1] = (1, 1)

    trip_offsets = np.array([0, 2, 4], dtype=np.int64)

    adj_offsets, adj_neighbors, adj_weights, adj_trip_ids = _build_adjacency(
        stop_times_array,
        trip_offsets,
        len(stop_ids),
    )

    (
        route_stop_offsets,
        route_stops,
        route_trip_offsets,
        route_trips,
        route_board_offsets,
        route_board_times,
        route_board_monotonic,
        stop_route_offsets,
        stop_routes,
    ) = _build_routes(stop_times_array, trip_offsets, len(stop_ids))

    stop_id_index = {value: idx for idx, value in enumerate(stop_ids)}
    trip_id_index = {value: idx for idx, value in enumerate(trip_ids)}
    stop_lats = np.array([46.5190, 46.5200, 46.5218], dtype=np.float64)
    stop_lons = np.array([6.6320, 6.6330, 6.6330], dtype=np.float64)
    transfer_offsets, transfer_neighbors, transfer_weights = _build_transfers(
        stop_ids,
        stop_lats,
        stop_lons,
    )
    trip_route_types = np.array([2, 3], dtype=np.int16)
    trip_cost_factors = _build_trip_cost_factors(trip_route_types)

    return TransitNetwork(
        stops=stops_array,
        stop_times=stop_times_array,
        routes=routes_array,
        stop_id_index=stop_id_index,
        trip_id_index=trip_id_index,
        stop_ids=stop_ids,
        stop_lats=stop_lats,
        stop_lons=stop_lons,
        trip_ids=trip_ids,
        trip_offsets=trip_offsets,
        adj_offsets=adj_offsets,
        adj_neighbors=adj_neighbors,
        adj_weights=adj_weights,
        adj_trip_ids=adj_trip_ids,
        route_stop_offsets=route_stop_offsets,
        route_stops=route_stops,
        route_trip_offsets=route_trip_offsets,
        route_trips=route_trips,
        route_board_offsets=route_board_offsets,
        route_board_times=route_board_times,
        route_board_monotonic=route_board_monotonic,
        stop_route_offsets=stop_route_offsets,
        stop_routes=stop_routes,
        trip_route_types=trip_route_types,
        trip_cost_factors=trip_cost_factors,
        transfer_offsets=transfer_offsets,
        transfer_neighbors=transfer_neighbors,
        transfer_weights=transfer_weights,
    )


def test_smoke_mock_network():
    network = build_mock_network()
    earliest, pred_stop, pred_trip, pred_time = run_raptor(
        network.stop_times,
        network.trip_offsets,
        network.route_stop_offsets,
        network.route_stops,
        network.route_trip_offsets,
        network.route_trips,
        network.route_board_offsets,
        network.route_board_times,
        network.route_board_monotonic,
        network.stop_route_offsets,
        network.stop_routes,
        network.trip_cost_factors,
        300,
        network.transfer_offsets,
        network.transfer_neighbors,
        network.transfer_weights,
        0,
        2,
        900,
    )
    segments = build_path(
        network.stop_times,
        network.trip_offsets,
        2,
        earliest,
        pred_stop,
        pred_trip,
        pred_time,
    )

    assert segments
    assert segments[-1][2] <= 1100


def test_smoke_mock_network_dijkstra():
    network = build_mock_network()
    dist, pred_stop, pred_trip = run_dijkstra_fast(
        network.adj_offsets,
        network.adj_neighbors,
        network.adj_weights,
        network.adj_trip_ids,
        0,
        2,
        900,
    )
    segments = build_path_dijkstra(
        2,
        dist,
        pred_stop,
        pred_trip,
    )

    assert segments
    assert segments[-1][2] == 990


def test_smoke_mock_network_astar():
    network = build_mock_network()
    heuristic = np.zeros(network.adj_offsets.shape[0] - 1, dtype=np.int64)
    dist, pred_stop, pred_trip = run_astar_fast(
        network.adj_offsets,
        network.adj_neighbors,
        network.adj_weights,
        network.adj_trip_ids,
        0,
        2,
        900,
        heuristic,
    )
    segments = build_path_dijkstra(
        2,
        dist,
        pred_stop,
        pred_trip,
    )

    assert segments
    assert segments[-1][2] == 990


def test_multi_departure_response_offsets():
    network = build_mock_network()
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["A"]],
        network.stop_id_index["C"],
        900,
    )

    assert response["options"]
    assert len(response["options"]) >= 1
    assert response["options"][0]["departure_time"] == 900
    assert response["options"][0]["segments"]
    assert "network_trip_profile" in response
    assert "trip_count" in response["network_trip_profile"]

    fixed = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["A"]],
        network.stop_id_index["C"],
        900,
        offset_minutes=(0, 10, 20, 30, 40),
    )
    assert len(fixed["options"]) == 5
    assert fixed["options"][1]["departure_time"] == 1500


def test_http_multi_start_selects_fastest_path():
    network = build_mock_network()
    response = build_multi_departure_response(
        network,
        "dijkstra",
        [network.stop_id_index["A"], network.stop_id_index["B"]],
        network.stop_id_index["C"],
        900,
        offset_minutes=(0,),
    )

    assert response["segments"]
    assert response["segments"][-1]["arrival_time"] == 940
    assert response["max_transfers"] == 4


def test_http_segments_include_stop_coordinates():
    network = build_mock_network()
    response = build_multi_departure_response(
        network,
        "dijkstra",
        [network.stop_id_index["A"]],
        network.stop_id_index["C"],
        900,
        offset_minutes=(0,),
    )

    assert response["segments"]
    segment = response["segments"][0]
    assert "lat" in segment and "lon" in segment
    assert isinstance(segment["lat"], float)
    assert isinstance(segment["lon"], float)


def test_transfer_segment_includes_walking_geometry():
    network = build_mock_network()
    start_idx = network.stop_id_index["A"]
    to_idx = network.stop_id_index["B"]
    segments = [(-2, to_idx, 930)]

    payloads = _build_segment_payloads(network, segments, start_idx, 900)

    assert len(payloads) == 1
    segment = payloads[0]
    assert segment["trip_id"] == "TRANSFER"
    assert segment["from_stop_id"] == "A"
    assert segment["walk_duration_seconds"] == 30
    assert "walking_geometry" in segment
    assert segment["walking_geometry"]["type"] == "LineString"
    assert len(segment["walking_geometry"]["coordinates"]) >= 2
    assert segment["walk_distance_m"] > 0


def test_http_origin_destination_includes_access_and_egress_walk_segments(monkeypatch):
    network = build_mock_network()

    # Keep geometry deterministic for the test by forcing straight-line fallback.
    monkeypatch.setattr("src.http_server._find_walking_path_via_osrm", lambda *args, **kwargs: None)

    start_indices, start_penalties, origin_metadata = _select_starts_from_origin(
        network,
        {"lat": 48.8566, "lon": 2.3522, "radius_m": 1200, "max_candidates": 3},
    )
    end_indices, end_penalties, destination_metadata = _select_ends_from_destination(
        network,
        {"lat": 48.8574, "lon": 2.3540, "radius_m": 1200, "max_candidates": 3},
    )

    response = build_multi_departure_response(
        network,
        "dijkstra",
        start_indices,
        end_indices,
        900,
        offset_minutes=(0,),
        start_penalties=start_penalties,
        end_penalties=end_penalties,
        metadata={**origin_metadata, **destination_metadata},
    )

    assert response["segments"]
    first_segment = response["segments"][0]
    last_segment = response["segments"][-1]

    assert first_segment["trip_id"] == "TRANSFER"
    assert first_segment["walking_segment_type"] == "access"
    assert first_segment["from_stop_id"] == "ORIGIN"

    assert last_segment["trip_id"] == "TRANSFER"
    assert last_segment["walking_segment_type"] == "egress"
    assert last_segment["stop_id"] == "DESTINATION"


def test_http_departure_parses_numeric_string_timestamp():
    ts = 1738580100
    expected = datetime.fromtimestamp(ts)
    expected_seconds = expected.hour * 3600 + expected.minute * 60 + expected.second
    assert _departure_to_seconds(str(ts)) == expected_seconds


def test_http_raptor_returns_no_path_when_schedule_unavailable():
    network = _build_linear_transfer_network(14)
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["S0"]],
        network.stop_id_index["S13"],
        5000,
        offset_minutes=(0,),
    )

    assert response["segments"] == []


def test_origin_selects_candidate_starts_and_penalties():
    network = build_mock_network()
    start_indices, penalties, metadata = _select_starts_from_origin(
        network,
        {"lat": 48.8566, "lon": 2.3522, "radius_m": 300, "max_candidates": 2, "seed_candidates": 2},
    )

    assert start_indices
    assert len(start_indices) <= 2
    assert isinstance(metadata, dict)
    assert metadata["candidate_start_count"] == len(start_indices)
    assert network.stop_id_index["A"] in start_indices
    assert penalties[network.stop_id_index["A"]] <= 15


def test_http_origin_route_without_explicit_start_ids():
    network = build_mock_network()
    start_indices, penalties, metadata = _select_starts_from_origin(
        network,
        {"lat": 48.8566, "lon": 2.3522, "radius_m": 1200, "max_candidates": 3},
    )
    response = build_multi_departure_response(
        network,
        "dijkstra",
        start_indices,
        network.stop_id_index["C"],
        900,
        offset_minutes=(0,),
        start_penalties=penalties,
        metadata=metadata,
    )

    assert response["segments"]
    assert response["start_stop_id"] in network.stop_id_index
    assert response["candidate_start_count"] == len(start_indices)


def test_http_destination_selects_candidate_ends_and_penalties():
    network = build_mock_network()
    end_indices, penalties, metadata = _select_ends_from_destination(
        network,
        {"lat": 48.8574, "lon": 2.3540, "radius_m": 300, "max_candidates": 2},
    )

    assert end_indices
    assert len(end_indices) <= 2
    assert isinstance(metadata, dict)
    assert end_indices[0] == network.stop_id_index["C"]
    assert penalties[end_indices[0]] <= 15


def test_http_origin_and_destination_route_without_explicit_stop_ids():
    network = build_mock_network()
    start_indices, start_penalties, origin_metadata = _select_starts_from_origin(
        network,
        {"lat": 48.8566, "lon": 2.3522, "radius_m": 1200, "max_candidates": 3},
    )
    end_indices, end_penalties, destination_metadata = _select_ends_from_destination(
        network,
        {"lat": 48.8574, "lon": 2.3540, "radius_m": 1200, "max_candidates": 3},
    )

    response = build_multi_departure_response(
        network,
        "dijkstra",
        start_indices,
        end_indices,
        900,
        offset_minutes=(0,),
        start_penalties=start_penalties,
        end_penalties=end_penalties,
        metadata={**origin_metadata, **destination_metadata},
    )

    assert response["segments"]
    assert response["start_stop_id"] in network.stop_id_index
    assert response["end_stop_id"] in network.stop_id_index
    assert response["candidate_start_count"] == len(start_indices)
    assert response["candidate_end_count"] == len(end_indices)


def test_get_start_stop_ids_prefers_repeated_field():
    request = SimpleNamespace(start_stop_id="A", start_stop_ids=[" B ", "A", "", "B"])
    assert _get_start_stop_ids(request) == ["B", "A"]


def test_resolve_stop_query_exact_stop_id():
    network = build_mock_network()
    idx, meta = _resolve_stop_query_to_index(network, "A")

    assert idx == network.stop_id_index["A"]
    assert meta is not None
    assert meta["match_type"] == "exact_stop_id"


def test_resolve_stop_query_fuzzy_with_stop_names():
    network = build_mock_network()
    network.stop_ids = ["STOP_LAUSANNE", "STOP_OURS", "STOP_ECHALLENS"]
    network.stop_id_index = {value: idx for idx, value in enumerate(network.stop_ids)}
    network.stop_names = np.asarray(["Lausanne Gare", "Lausanne Ours", "Echallens Gare"], dtype=object)

    idx, meta = _resolve_stop_query_to_index(network, "lausanne our")

    assert idx == 1
    assert meta is not None
    assert meta["match_type"] == "fuzzy"


def test_resolve_stop_query_returns_none_for_unknown():
    network = build_mock_network()
    idx, meta = _resolve_stop_query_to_index(network, "zzzz impossible stop")

    assert idx is None
    assert meta is None


def test_parallel_multi_start_selects_fastest_path():
    network = build_mock_network()
    start_a = network.stop_id_index["A"]
    start_b = network.stop_id_index["B"]
    end_c = network.stop_id_index["C"]

    segments = asyncio.run(
        _find_fastest_segments_parallel(
            network,
            "dijkstra",
            [start_a, start_b],
            end_c,
            900,
        )
    )

    assert segments
    assert segments[-1][2] == 940


def test_raptor_handles_long_transfer_chain():
    network = _build_linear_transfer_network(14)
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["S0"]],
        network.stop_id_index["S13"],
        900,
        offset_minutes=(0,),
        max_transfers=20,
    )

    assert response["segments"]
    assert response["segments"][-1]["stop_id"] == "S13"


def test_raptor_option_exposes_resolver_metadata():
    network = _build_linear_transfer_network(14)
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["S0"]],
        network.stop_id_index["S13"],
        900,
        offset_minutes=(0,),
    )

    option = response["options"][0]
    assert "resolver_algorithm" in option
    assert "fallback_used" in option


def test_raptor_prefers_train_over_bus_when_weighted_cost_is_lower():
    network = _build_mode_preference_network()
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["A"]],
        network.stop_id_index["C"],
        900,
        offset_minutes=(0,),
    )

    assert response["segments"]
    assert response["segments"][-1]["trip_id"] == "TRAIN_DIRECT"
    itinerary_trip_profile = response["options"][0]["itinerary_trip_profile"]
    assert itinerary_trip_profile["distinct_trip_count"] >= 1
    assert itinerary_trip_profile["route_type_counts"]


def test_raptor_prefers_shuttle_bus_over_short_walk_shortcut():
    network = _build_short_walk_vs_shuttle_network()
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["A"]],
        network.stop_id_index["C"],
        850,
        offset_minutes=(0,),
    )

    assert response["segments"]
    assert response["segments"][-1]["arrival_time"] == 1200
    assert not any(segment["trip_id"] == "TRANSFER" for segment in response["segments"])


def test_transfer_cap_rejects_high_transfer_path():
    network = _build_linear_transfer_network(14)
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["S0"]],
        network.stop_id_index["S13"],
        900,
        offset_minutes=(0,),
        max_transfers=2,
    )

    assert response["segments"] == []
    assert response["max_transfers"] == 2


def test_raptor_option_exposes_diagnostics_on_no_path():
    network = _build_linear_transfer_network(14)
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["S0"]],
        network.stop_id_index["S13"],
        5000,
        offset_minutes=(0,),
    )

    diagnostics = response["options"][0]["raptor_diagnostics"]
    assert diagnostics is not None
    assert diagnostics["attempt_caps"]
    assert "no_path_reason" in diagnostics


def test_raptor_adaptive_destination_expansion_finds_path():
    network = _build_destination_trap_network()
    start_indices, start_penalties, origin_metadata = _select_starts_from_origin(
        network,
        {"lat": 46.5, "lon": 6.6, "radius_m": 100, "max_candidates": 1},
    )
    end_indices, end_penalties, destination_metadata = _select_ends_from_destination(
        network,
        {"lat": 46.5260, "lon": 6.6260, "radius_m": 5, "max_candidates": 2, "seed_candidates": 1},
    )

    response = build_multi_departure_response(
        network,
        "raptor",
        start_indices,
        end_indices,
        900,
        offset_minutes=(0,),
        start_penalties=start_penalties,
        end_penalties=end_penalties,
        metadata={**origin_metadata, **destination_metadata},
    )

    assert response["segments"]
    diagnostics = response["options"][0]["raptor_diagnostics"]
    assert diagnostics is not None
    assert 1 in diagnostics["attempted_end_candidates"]
    assert 2 in diagnostics["attempted_end_candidates"]
