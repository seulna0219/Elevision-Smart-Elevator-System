#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo5_emergency.py – 兩部電梯「急救直達模式」示範

- 容量上限 15 人
- 20 層樓
- 電梯 A 起始 3F；電梯 B 起始 14F
- 急救情境：
    1. 10F 有病患，急救人員先到 1F。A 從 3F → 1F 載急救人員（1 人，目的地 10F）。
    2. A 直達 10F，開門 → 10 分鐘救援。
    3. 救援結束後，門再關，病患（1 人）上車，目的地 1F。
    4. A 直達 1F，下客 (急救人員 + 病患)。
- 電梯 B 保持空載 (示範用)
"""

import time
import threading
import random

# --- 參數 ---
total_floors    = 20
MOVE_DELAY      = 0.5
DOOR_OPEN_TIME  = 2.0
DOOR_CLOSE_TIME = 2.0
PAX_MIN_TIME    = 1
PAX_MAX_TIME    = 5
REScUE_TIME     = 600   # 10 分鐘救援 (秒)
COL_WIDTH       = 32

class Elevator:
    def __init__(self, eid, start):
        self.id = eid
        self.current_floor = start
        self.elapsed = 0.0
        self.load = 0

class Controller:
    def __init__(self, elevators):
        self.elevators = {e.id: e for e in elevators}
        self.plock = threading.Lock()

    def _print(self, car, action):
        msg = f"[{car.id}] {action}"
        with self.plock:
            if car.id == 'A':
                print(f"{msg:<{COL_WIDTH}}")
            else:
                print(f"{'':<{COL_WIDTH}}{msg}")

    def _travel(self, car, dest):
        step = 1 if dest > car.current_floor else -1
        while car.current_floor != dest:
            car.current_floor += step
            time.sleep(MOVE_DELAY); car.elapsed += MOVE_DELAY
            self._print(car, f"Move to floor {car.current_floor}")

    def _open_close(self, car, label, dests=None, unload=False):
        # 開門
        self._print(car, "Opening door")
        time.sleep(DOOR_OPEN_TIME); car.elapsed += DOOR_OPEN_TIME

        # 上客 (若有)
        if label and dests:
            total_board_time = 0
            for _ in dests:
                bt = random.randint(PAX_MIN_TIME, PAX_MAX_TIME)
                time.sleep(bt); car.elapsed += bt; total_board_time += bt
            car.load += len(dests)
            self._print(car, f"Boarded {len(dests)} pax -> {dests} (spent {total_board_time} sec.)")

        # 下客 (一次一人)
        if unload:
            ot = random.randint(PAX_MIN_TIME, PAX_MAX_TIME)
            time.sleep(ot); car.elapsed += ot; car.load -= 1
            self._print(car, f"Get off 1 pax at {car.current_floor} (spent {ot} sec.)")

        # 關門
        self._print(car, "Closing door")
        time.sleep(DOOR_CLOSE_TIME); car.elapsed += DOOR_CLOSE_TIME

    def _rescue(self, car, duration):
        # 救援過程
        self._print(car, f"Rescue in progress ({duration} sec.)")
        time.sleep(duration); car.elapsed += duration
        self._print(car, "Rescue complete")

if __name__ == '__main__':
    # 初始化電梯位置
    A = Elevator('A', 3)
    B = Elevator('B', 14)
    ctl = Controller([A, B])

    print(f"Start → A:{A.current_floor}  B:{B.current_floor}\n")

    # 電梯 A 路線：急救直達模式
    route_A = [
        # 1) 1F 載急救人員 1 人 (目的地 10F)
        (1, 'RescueTeam', [10], False),
        # 2) 10F 救援
        (10, 'Rescue', None, False),
        # 3) 10F 載病患 1 人 (目的地 1F)
        (10, 'Patient', [1]*1, False),
        # 4) 1F 下客急救人員 + 病患 x2
        (1, None, None, True),
        (1, None, None, True),
    ]

    # 電梯 B 保持空載 (示範用)
    route_B = []

    # 執行並行運行
    def run_route(car, route):
        for dest, label, dests, unload in route:
            ctl._travel(car, dest)
            if label == 'Rescue':
                ctl._rescue(car, REScUE_TIME)
            else:
                ctl._open_close(car, label, dests, unload)

    tA = threading.Thread(target=run_route, args=(A, route_A))
    tB = threading.Thread(target=run_route, args=(B, route_B))
    tA.start(); tB.start()
    tA.join(); tB.join()

    print("\n----- Result -----")
    print(f"Elevator A run time : {A.elapsed:.1f} s")
    print(f"Elevator B run time : {B.elapsed:.1f} s")
