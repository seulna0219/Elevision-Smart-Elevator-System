import random, copy
from collections import defaultdict

RANDOM_SEED    = 42 #None or 42
MOVE_PER_FLOOR = 0.5
DOOR_OPEN      = 2.0
DOOR_CLOSE     = 2.0
CAP_SPACE      = 8
DEST_FLOOR     = 1

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def rnd_time():
    return float(random.randint(1, 5)) 

class Passenger:
    def __init__(self, kind="normal", space=1):
        self.kind = kind
        self.space = space
        self.board_t = rnd_time()
        self.alight_t = rnd_time()

class BaseElev:
    def __init__(self, name, start_floor):
        self.name = name
        self.floor = start_floor
        self.space_used = 0
        self.onboard = []
        self.total_time = 0.0
        self.trip_stats = []  
        self.logs = []

    def move_to(self, target_floor):
        if target_floor == self.floor:
            return 0.0
        dist = abs(self.floor - target_floor)
        t = dist * MOVE_PER_FLOOR
        self.total_time += t
        self.floor = target_floor
        return t

    def door_cycle(self):
        t = DOOR_OPEN + DOOR_CLOSE
        self.total_time += t
        return t

    def board_until_full(self, floor_list):
        boarded = 0
        while floor_list and (self.space_used + floor_list[0].space) <= CAP_SPACE:
            p = floor_list.pop(0)
            self.onboard.append(p)
            self.space_used += p.space
            self.total_time += p.board_t
            boarded += 1
        return boarded

    def alight_all_at_1F(self):
        if not self.onboard:
            return 0
        cnt = 0
        while self.onboard:
            p = self.onboard.pop(0)
            self.total_time += p.alight_t
            cnt += 1
        self.space_used = 0
        return cnt

class ElevTraditional(BaseElev):
    def run_plan(self, floors, plan):
        stops, t0 = 0, self.total_time
        for f in plan:
            self.move_to(f)
            self.door_cycle(); stops += 1
            if f == DEST_FLOOR:
                self.alight_all_at_1F()
            else:
                before = len(floors[f])
                boarded = self.board_until_full(floors[f])
        self.trip_stats.append((stops, self.total_time - t0))

class ElevCV(BaseElev):
    def is_full(self): return self.space_used >= CAP_SPACE
    def run_plan(self, floors, plan):
        stops, t0 = 0, self.total_time
        i = 0
        while i < len(plan):
            f = plan[i]
            if self.is_full() and f != DEST_FLOOR:
                self.move_to(DEST_FLOOR)
                self.door_cycle(); stops += 1
                self.alight_all_at_1F()
                break

            self.move_to(f)
            self.door_cycle(); stops += 1

            if f == DEST_FLOOR:
                self.alight_all_at_1F()
            else:
                self.board_until_full(floors[f])
            i += 1

        if i == len(plan) and self.floor != DEST_FLOOR and self.onboard:
            self.move_to(DEST_FLOOR)
            self.door_cycle(); stops += 1
            self.alight_all_at_1F()
        self.trip_stats.append((stops, self.total_time - t0))

def make_initial_floors():
    floors = defaultdict(list)
    def add(f, n, kind="normal", space=1):
        for _ in range(n): floors[f].append(Passenger(kind, space))
    add(20, 2, "cleaner", 4)
    add(15, 3); add(13, 2); add(12, 1); add(11, 2); add(10, 2)
    add(9,  2, "cleaner", 4)
    add(8, 2); add(7, 2); add(6, 1); add(5, 2); add(4, 1); add(2, 1)
    return floors

def snapshot_counts(floors, watch):
    return {f: len(floors.get(f, [])) for f in watch}

def print_table(title, rows, headers):
    print("\n" + title)
    print("-" * len(title))
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-"*w for w in widths))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))

def run_both_systems():
    base = make_initial_floors()
    floors_trad = copy.deepcopy(base)
    floors_cv   = copy.deepcopy(base)

    #路線（固定）
    A_trip1 = [20,15,13,12,11,10,1]
    A_trip2 = [15,13,12,11,10,1]
    A_trip3 = [10,1]

    B_trip1 = [9,8,7,6,5,4,2,1]
    B_trip2_trad = [8,7,6,5,4,2,1]
    B_trip2_cv   = [8,7,6,5,4,1]
    B_trip3 = [2,1]

    watch_A = [20,15,13,12,11,10]
    watch_B = [9,8,7,6,5,4,2]

    #傳統
    A_t = ElevTraditional("A", 16); B_t = ElevTraditional("B", 9)
    A_t_snaps = [snapshot_counts(floors_trad, watch_A)]
    B_t_snaps = [snapshot_counts(floors_trad, watch_B)]

    A_t.run_plan(floors_trad, A_trip1); A_t_snaps.append(snapshot_counts(floors_trad, watch_A))
    A_t.run_plan(floors_trad, A_trip2); A_t_snaps.append(snapshot_counts(floors_trad, watch_A))
    A_t.run_plan(floors_trad, A_trip3); A_t_snaps.append(snapshot_counts(floors_trad, watch_A))

    B_t.run_plan(floors_trad, B_trip1); B_t_snaps.append(snapshot_counts(floors_trad, watch_B))
    B_t.run_plan(floors_trad, B_trip2_trad); B_t_snaps.append(snapshot_counts(floors_trad, watch_B))
    B_t.run_plan(floors_trad, B_trip3); B_t_snaps.append(snapshot_counts(floors_trad, watch_B))

    print_table(
        "傳統電梯系統：電梯A運行情況",
        [[f, A_t_snaps[0][f], A_t_snaps[1][f], A_t_snaps[2][f], A_t_snaps[3][f]] for f in watch_A],
        headers=["樓層","起始每層人數","第1趟後剩餘","第2趟後剩餘","第3趟後剩餘"]
    )
    print_table(
        "傳統電梯系統：電梯Ｂ運行情況",
        [[f, B_t_snaps[0][f], B_t_snaps[1][f], B_t_snaps[2][f], B_t_snaps[3][f]] for f in watch_B],
        headers=["樓層","起始每層人數","第1趟後剩餘","第2趟後剩餘","第3趟後剩餘"]
    )

    A1s,A1t = A_t.trip_stats[0]; A2s,A2t = A_t.trip_stats[1]; A3s,A3t = A_t.trip_stats[2]
    B1s,B1t = B_t.trip_stats[0]; B2s,B2t = B_t.trip_stats[1]; B3s,B3t = B_t.trip_stats[2]
    total_trad_stops = A1s+A2s+A3s+B1s+B2s+B3s
    total_trad_time  = A1t+A2t+A3t+B1t+B2t+B3t

    print_table(
        "傳統電梯系統：運行效率",
        [
            ["電梯 A 第1趟", A1s, f"{A1t:.1f}"],
            ["電梯 A 第2趟", A2s, f"{A2t:.1f}"],
            ["電梯 A 第3趟", A3s, f"{A3t:.1f}"],
            ["電梯 B 第1趟", B1s, f"{B1t:.1f}"],
            ["電梯 B 第2趟", B2s, f"{B2t:.1f}"],
            ["電梯 B 第3趟", B3s, f"{B3t:.1f}"],
            ["合計", total_trad_stops, f"{total_trad_time:.1f}"],
        ],
        headers=["參數","一趟停靠次數","一趟運行時 / s"]
    )

    # ===== 專題（機器視覺）=====
    floors_cv = copy.deepcopy(base)  # 同一批乘客，公平比較
    A_c = ElevCV("A", 16); B_c = ElevCV("B", 9)

    A_c_snaps = [snapshot_counts(floors_cv, watch_A)]
    B_c_snaps = [snapshot_counts(floors_cv, watch_B)]

    A_c.run_plan(floors_cv, A_trip1); A_c_snaps.append(snapshot_counts(floors_cv, watch_A))
    A_c.run_plan(floors_cv, A_trip2); A_c_snaps.append(snapshot_counts(floors_cv, watch_A))
    A_c.run_plan(floors_cv, A_trip3); A_c_snaps.append(snapshot_counts(floors_cv, watch_A))

    B_c.run_plan(floors_cv, B_trip1);      B_c_snaps.append(snapshot_counts(floors_cv, watch_B))
    B_c.run_plan(floors_cv, B_trip2_cv);   B_c_snaps.append(snapshot_counts(floors_cv, watch_B))
    B_c.run_plan(floors_cv, B_trip3);      B_c_snaps.append(snapshot_counts(floors_cv, watch_B))

    print_table(
        "智慧電梯系統：電梯A運行情況",
        [[f, A_c_snaps[0][f], A_c_snaps[1][f], A_c_snaps[2][f], A_c_snaps[3][f]] for f in watch_A],
        headers=["樓層","起始每層人數","第1趟後剩餘","第2趟後剩餘","第3趟後剩餘"]
    )
    print_table(
        "智慧電梯系統：電梯Ｂ運行情況",
        [[f, B_c_snaps[0][f], B_c_snaps[1][f], B_c_snaps[2][f], B_c_snaps[3][f]] for f in watch_B],
        headers=["樓層","起始每層人數","第1趟後剩餘","第2趟後剩餘","第3趟後剩餘"]
    )

    A1s,A1t = A_c.trip_stats[0]; A2s,A2t = A_c.trip_stats[1]; A3s,A3t = A_c.trip_stats[2]
    B1s,B1t = B_c.trip_stats[0]; B2s,B2t = B_c.trip_stats[1]; B3s,B3t = B_c.trip_stats[2]
    total_cv_stops = A1s+A2s+A3s+B1s+B2s+B3s
    total_cv_time  = A1t+A2t+A3t+B1t+B2t+B3t

    print_table(
        "智慧電梯系統：運行效率",
        [
            ["電梯 A 第1趟", A1s, f"{A1t:.1f}"],
            ["電梯 A 第2趟", A2s, f"{A2t:.1f}"],
            ["電梯 A 第3趟", A3s, f"{A3t:.1f}"],
            ["電梯 B 第1趟", B1s, f"{B1t:.1f}"],
            ["電梯 B 第2趟", B2s, f"{B2t:.1f}"],
            ["電梯 B 第3趟", B3s, f"{B3t:.1f}"],
            ["合計", total_cv_stops, f"{total_cv_time:.1f}"],
        ],
        headers=["參數","一趟停靠次數","一趟運行時 / s"]
    )

    # ===== 總結 =====
    dt = total_trad_time - total_cv_time
    ds = total_trad_stops - total_cv_stops
    pct_t = (dt/total_trad_time*100.0) if total_trad_time>0 else 0.0
    pct_s = (ds/total_trad_stops*100.0) if total_trad_stops>0 else 0.0

    print("\n=== 總結 ===")
    print(f"停靠：傳統電梯系統 {total_trad_stops} 次 → 智慧電梯系統 {total_cv_stops} 次，改善 {ds} 次（{pct_s:.1f}%）")
    print(f"時間：傳統電梯系統 {total_trad_time:.1f}s → 智慧電梯系統 {total_cv_time:.1f}s，節省 {dt:.1f}s（{pct_t:.1f}%）")

if __name__ == "__main__":
    run_both_systems()
