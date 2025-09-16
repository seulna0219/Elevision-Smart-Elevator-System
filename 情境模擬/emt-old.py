#傳統電梯系統

import random
from dataclasses import dataclass, field
from typing import List, Tuple

# ===== 參數（可調）=====
FLOORS = 20
EMERGENCY_FLOOR = 10
START_FLOOR = 1

INIT_FLOOR_A = 3
INIT_FLOOR_B = 16

SPEED_PER_FLOOR = 0.5
DOOR_OPEN = 2.0
DOOR_CLOSE = 2.0
PERSON_TIME_RANGE = (1, 5)
RESCUE_TIME = 10 * 60

NUM_INCIDENTAL_UP = 4
NUM_INCIDENTAL_DOWN = 3

MAX_BOARD_UP = 3
MAX_ALIGHT_UP = 3
MAX_BOARD_DOWN = 2
MAX_ALIGHT_DOWN = 2

PARAMEDIC_COUNT = 1
PATIENT_COUNT = 1

STAIR_UP_PER_FLOOR = 8.0

STOP_ON_CALL_NO_BOARD_WHEN_EMERGENCY = True
RANDOM_SEED = 42


@dataclass
class Event:
    timestamp: float
    description: str

@dataclass
class SimulationResult:
    log: List[Event] = field(default_factory=list)
    time_to_arrive: float = 0.0
    time_rescue: float = 0.0
    time_return: float = 0.0
    total_time: float = 0.0
    chosen_strategy: str = ""


def sec_fmt(s: float) -> str:
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, r = divmod(r, 60)
    return f"{h:d}:{m:02d}:{r:02d}" if h else f"{m:d}:{r:02d}"

def add_log(log: List[Event], t: float, msg: str):
    log.append(Event(timestamp=t, description=msg))

def travel_time(f1: int, f2: int) -> float:
    return abs(f2 - f1) * SPEED_PER_FLOOR

def rand_person_time(n: int) -> float:
    return sum(random.randint(*PERSON_TIME_RANGE) for _ in range(n))

def pick_incidental_stops(start: int, end: int, count: int) -> List[int]:
    if start < end:
        candidates = list(range(start + 1, end))
    else:
        candidates = list(range(end + 1, start))
    if not candidates or count <= 0:
        return []
    count = min(count, len(candidates))
    return sorted(random.sample(candidates, count))

def nearest_elevator_time_to_floor(elev_positions: dict, target_floor: int) -> Tuple[str, float]:
    best_name, best_time = None, float("inf")
    for name, pos in elev_positions.items():
        t = travel_time(pos, target_floor)
        if t < best_time:
            best_time = t
            best_name = name
    return best_name, best_time


def simulate_traditional(strategy: str, post_rescue_leave: bool, seed: int = RANDOM_SEED) -> SimulationResult:
    """
    strategy: "wait_elevator" or "take_stairs"
    post_rescue_leave: True = 電梯離開，False = 電梯原地等待
    """
    if seed is not None:
        random.seed(seed)

    log: List[Event] = []
    t = 0.0
    elev_pos = {"A": INIT_FLOOR_A, "B": INIT_FLOOR_B}

    add_log(log, t, f"模擬開始。A在{INIT_FLOOR_A}樓，B在{INIT_FLOOR_B}樓。緊急樓層：{EMERGENCY_FLOOR}。")
    add_log(log, t, f"策略：{'等待電梯' if strategy=='wait_elevator' else '走樓梯'}；救援後電梯{'離開' if post_rescue_leave else '原地等待'}。")

    onboard_regular = 0
    onboard_special = 0
    emergency_onboard = False

    if strategy == "wait_elevator":
        chosen, travel_t = nearest_elevator_time_to_floor(elev_pos, START_FLOOR)
        t += travel_t
        add_log(log, t, f"電梯{chosen}自{elev_pos[chosen]}樓抵達1樓（{travel_t:.1f}s）。")
        elev_pos[chosen] = START_FLOOR

        # 1F 上車
        board = PARAMEDIC_COUNT
        dwell = DOOR_OPEN + rand_person_time(board) + DOOR_CLOSE
        t += dwell
        onboard_special += board
        emergency_onboard = True
        add_log(log, t, f"1樓開門：進來{board}人，出來0人（耗時 {dwell:.1f}s）。")

        # 上行
        incidental_up = pick_incidental_stops(START_FLOOR, EMERGENCY_FLOOR, NUM_INCIDENTAL_UP)
        for src, dst in zip([START_FLOOR]+incidental_up, incidental_up+[EMERGENCY_FLOOR]):
            tt = travel_time(src, dst)
            t += tt
            add_log(log, t, f"移動：{src} → {dst}（{tt:.1f}s）")
            elev_pos[chosen] = dst
            if dst != EMERGENCY_FLOOR and emergency_onboard and STOP_ON_CALL_NO_BOARD_WHEN_EMERGENCY:
                dwell = DOOR_OPEN + DOOR_CLOSE
                t += dwell
                add_log(log, t, f"{dst}樓（禮讓急救）：開門無人進出（耗時 {dwell:.1f}s）。")

        # 到站救援
        dwell = DOOR_OPEN + rand_person_time(onboard_special) + DOOR_CLOSE
        t += dwell
        onboard_special = 0
        emergency_onboard = False
        add_log(log, t, f"{EMERGENCY_FLOOR}樓開門（救援）：進來0人，出來{PARAMEDIC_COUNT}人（耗時 {dwell:.1f}s）。")

        time_to_arrive = t
        t += RESCUE_TIME
        add_log(log, t, f"救援完成（耗時 {sec_fmt(RESCUE_TIME)}）。")

        # 如果電梯離開 → 等它回來
        if post_rescue_leave:
            elev_pos[chosen] = random.randint(1, FLOORS)  # 模擬離開到任意樓層
            travel_back = travel_time(elev_pos[chosen], EMERGENCY_FLOOR)
            t += travel_back
            add_log(log, t, f"電梯{chosen}從{elev_pos[chosen]}樓返回{EMERGENCY_FLOOR}樓（{travel_back:.1f}s）。")
            elev_pos[chosen] = EMERGENCY_FLOOR

        # 病患+急救上車
        board = PARAMEDIC_COUNT + PATIENT_COUNT
        dwell = DOOR_OPEN + rand_person_time(board) + DOOR_CLOSE
        t += dwell
        onboard_special = board
        emergency_onboard = True
        add_log(log, t, f"{EMERGENCY_FLOOR}樓開門（病患上車）：進來{board}人，出來0人（耗時 {dwell:.1f}s）。")

        # 下行
        t_before_return = t
        incidental_down = pick_incidental_stops(EMERGENCY_FLOOR, START_FLOOR, NUM_INCIDENTAL_DOWN)
        for src, dst in zip([EMERGENCY_FLOOR]+incidental_down, incidental_down+[START_FLOOR]):
            tt = travel_time(src, dst)
            t += tt
            add_log(log, t, f"移動：{src} → {dst}（{tt:.1f}s）")
            elev_pos[chosen] = dst
            if dst != START_FLOOR and emergency_onboard and STOP_ON_CALL_NO_BOARD_WHEN_EMERGENCY:
                dwell = DOOR_OPEN + DOOR_CLOSE
                t += dwell
                add_log(log, t, f"{dst}樓（禮讓急救）：開門無人進出（耗時 {dwell:.1f}s）。")

        dwell = DOOR_OPEN + rand_person_time(onboard_special) + DOOR_CLOSE
        t += dwell
        onboard_special = 0
        emergency_onboard = False
        add_log(log, t, f"1樓開門（任務結束）：進來0人，出來{PARAMEDIC_COUNT+PATIENT_COUNT}人（耗時 {dwell:.1f}s）。")

        time_return = t - t_before_return
        total_time = t

    else:
        # 走樓梯
        stair_time = (EMERGENCY_FLOOR - START_FLOOR) * STAIR_UP_PER_FLOOR
        t += stair_time
        add_log(log, t, f"急救人員以樓梯抵達{EMERGENCY_FLOOR}樓（耗時 {stair_time:.1f}s）。")

        time_to_arrive = t
        t += RESCUE_TIME
        add_log(log, t, f"救援完成（耗時 {sec_fmt(RESCUE_TIME)}）。")

        chosen, travel_t = nearest_elevator_time_to_floor(elev_pos, EMERGENCY_FLOOR)
        t += travel_t
        add_log(log, t, f"電梯{chosen}自{elev_pos[chosen]}樓抵達{EMERGENCY_FLOOR}樓（{travel_t:.1f}s）。")

        board = PARAMEDIC_COUNT + PATIENT_COUNT
        dwell = DOOR_OPEN + rand_person_time(board) + DOOR_CLOSE
        t += dwell
        onboard_special = board
        emergency_onboard = True
        add_log(log, t, f"{EMERGENCY_FLOOR}樓開門（病患上車）：進來{board}人，出來0人（耗時 {dwell:.1f}s）。")

        t_before_return = t
        incidental_down = pick_incidental_stops(EMERGENCY_FLOOR, START_FLOOR, NUM_INCIDENTAL_DOWN)
        for src, dst in zip([EMERGENCY_FLOOR]+incidental_down, incidental_down+[START_FLOOR]):
            tt = travel_time(src, dst)
            t += tt
            add_log(log, t, f"移動：{src} → {dst}（{tt:.1f}s）")
            elev_pos[chosen] = dst
            if dst != START_FLOOR and emergency_onboard and STOP_ON_CALL_NO_BOARD_WHEN_EMERGENCY:
                dwell = DOOR_OPEN + DOOR_CLOSE
                t += dwell
                add_log(log, t, f"{dst}樓（禮讓急救）：開門無人進出（耗時 {dwell:.1f}s）。")

        dwell = DOOR_OPEN + rand_person_time(onboard_special) + DOOR_CLOSE
        t += dwell
        onboard_special = 0
        emergency_onboard = False
        add_log(log, t, f"1樓開門（任務結束）：進來0人，出來{PARAMEDIC_COUNT+PATIENT_COUNT}人（耗時 {dwell:.1f}s）。")

        time_return = t - t_before_return
        total_time = t

    return SimulationResult(log, time_to_arrive, RESCUE_TIME, time_return, total_time, strategy)


if __name__ == "__main__":
    scenarios = [
        ("wait_elevator", False),  # 原地等待
        ("wait_elevator", True),   # 離開再回來
        ("take_stairs", False)     # 走樓梯
    ]
    for strategy, leave in scenarios:
        res = simulate_traditional(strategy, leave)
        title = f"{'等待電梯' if strategy=='wait_elevator' else '走樓梯'} - 救援後{'離開' if leave else '原地'}"
        print(f"\n===== 傳統電梯 - {title} =====")
        for e in res.log:
            print(f"[t={sec_fmt(e.timestamp)}] {e.description}")
        print("\n===== 摘要 =====")
        print(f"1F→{EMERGENCY_FLOOR}F 抵達時間：{sec_fmt(res.time_to_arrive)}")
        print(f"現場救援時間：{sec_fmt(res.time_rescue)}")
        print(f"{EMERGENCY_FLOOR}F→1F 返回時間：{sec_fmt(res.time_return)}")
        print(f"總任務時間：{sec_fmt(res.total_time)}")
