#專題電梯系統

import random
from dataclasses import dataclass, field
from typing import List

# ===== 參數（可調）=====
FLOORS = 20
EMERGENCY_FLOOR = 10
START_FLOOR = 1

# 兩部電梯起始樓層（A 會被系統保留給急救任務，B 不影響任務）
INIT_FLOOR_A = 3
INIT_FLOOR_B = 16  # 僅紀錄，不參與任務

SPEED_PER_FLOOR = 0.5       # 每層移動秒數
DOOR_OPEN = 2.0
DOOR_CLOSE = 2.0
PERSON_TIME_RANGE = (1, 5)  # 單人進/出時間(秒)
RESCUE_TIME = 10 * 60       # 10 分鐘

# 人數
PARAMEDIC_COUNT = 1
PATIENT_COUNT = 1

# 偵測與派車延遲（入口監視器偵測到→A接到指令→開始動作）
DETECTION_TO_DISPATCH_DELAY = 0.0  # 秒，若要更保守可設 2~5 秒

RANDOM_SEED = 42  # 想要每次不同可設 None


# ===== 資料結構 =====
@dataclass
class Event:
    timestamp: float
    description: str

@dataclass
class SimulationResult:
    log: List[Event] = field(default_factory=list)
    time_to_arrive: float = 0.0   # 急救人員抵達 10F（開始救援）的時間點
    time_rescue: float = 0.0
    time_return: float = 0.0
    total_time: float = 0.0


# ===== 小工具 =====
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


# ===== 專題電梯系統：全程專用、不受乘客影響 =====
def simulate_project_elevator(seed: int = RANDOM_SEED) -> SimulationResult:
    if seed is not None:
        random.seed(seed)

    log: List[Event] = []
    t = 0.0

    # 電梯起始位置
    posA = INIT_FLOOR_A
    posB = INIT_FLOOR_B  # 不使用，但保留紀錄

    add_log(log, t, f"專題系統啟動：入口監視器偵測到急救人員。A在{posA}樓，B在{posB}樓。")

    # 偵測到→下達派車指令的延遲
    if DETECTION_TO_DISPATCH_DELAY > 0:
        t += DETECTION_TO_DISPATCH_DELAY
        add_log(log, t, f"系統派車延遲 {DETECTION_TO_DISPATCH_DELAY:.1f}s。")

    # A 自動下到 1F 等待
    if posA != START_FLOOR:
        tt = travel_time(posA, START_FLOOR)
        t += tt
        add_log(log, t, f"A：{posA} → {START_FLOOR}（{tt:.1f}s），到站待命。")
        posA = START_FLOOR
    else:
        add_log(log, t, f"A 已在 1F 待命。")

    # 1F：急救人員上車（一定開門）
    board = PARAMEDIC_COUNT
    alight = 0
    dwell = DOOR_OPEN + rand_person_time(board) + DOOR_CLOSE
    t += dwell
    add_log(log, t, f"1樓開門：進來{board}人，出來{alight}人（耗時 {dwell:.1f}s）。")

    # 直達 10F（不中停、無他人影響）
    tt = travel_time(START_FLOOR, EMERGENCY_FLOOR)
    t += tt
    add_log(log, t, f"A 直達：{START_FLOOR} → {EMERGENCY_FLOOR}（{tt:.1f}s）。")

    # 到站救援：急救人員下車（一定開門）
    board = 0
    alight = PARAMEDIC_COUNT
    dwell = DOOR_OPEN + rand_person_time(alight) + DOOR_CLOSE
    t += dwell
    add_log(log, t, f"{EMERGENCY_FLOOR}樓開門（到站救援）：進來{board}人，出來{alight}人（耗時 {dwell:.1f}s）。")

    # 記錄抵達救援時間（到站+下車開門動作完成即開始救援）
    time_to_arrive = t
    add_log(log, t, f"開始救援（耗時 {sec_fmt(RESCUE_TIME)}）。")

    # 救援中
    t += RESCUE_TIME

    # 救援完成：病患+急救人員上車（一定開門）
    board = PARAMEDIC_COUNT + PATIENT_COUNT
    alight = 0
    dwell = DOOR_OPEN + rand_person_time(board) + DOOR_CLOSE
    t += dwell
    add_log(log, t, f"{EMERGENCY_FLOOR}樓開門（救援完成上車）：進來{board}人，出來{alight}人（耗時 {dwell:.1f}s）。")

    # 直達 1F（不中停）
    t_before_return = t
    tt = travel_time(EMERGENCY_FLOOR, START_FLOOR)
    t += tt
    add_log(log, t, f"A 直達：{EMERGENCY_FLOOR} → {START_FLOOR}（{tt:.1f}s）。")

    # 1F：全部下車（一定開門）
    board = 0
    alight = PARAMEDIC_COUNT + PATIENT_COUNT
    dwell = DOOR_OPEN + rand_person_time(alight) + DOOR_CLOSE
    t += dwell
    add_log(log, t, f"1樓開門（任務結束）：進來{board}人，出來{alight}人（耗時 {dwell:.1f}s）。")

    time_return = t - t_before_return
    total_time = t

    return SimulationResult(
        log=log,
        time_to_arrive=time_to_arrive,
        time_rescue=RESCUE_TIME,
        time_return=time_return,
        total_time=total_time
    )


# ===== 執行與輸出 =====
if __name__ == "__main__":
    res = simulate_project_elevator()

    print("===== 專題電梯系統 - 緊急情境（A 自動下到 1F，全程專用）=====")
    for e in res.log:
        print(f"[t={sec_fmt(e.timestamp)}] {e.description}")

    print("\n===== 摘要 =====")
    print(f"1F→{EMERGENCY_FLOOR}F 抵達時間：{sec_fmt(res.time_to_arrive)}")
    print(f"現場救援時間：{sec_fmt(res.time_rescue)}")
    print(f"{EMERGENCY_FLOOR}F→1F 返回時間：{sec_fmt(res.time_return)}")
    print(f"總任務時間：{sec_fmt(res.total_time)}")
