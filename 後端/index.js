const { MongoClient } = require('mongodb');

const uri = 'mongodb://127.0.0.1:27017';
const client = new MongoClient(uri);

async function main() {
  try {
    await client.connect();
    console.log('✅ 已連線到 MongoDB');

    const db = client.db("elevator_emergency_dbb");

    // 1. Users Collection
    const users = db.collection("users");
    await users.insertMany([
      {
        username: "admin",
        password: "password123", // 建議改成加密
        role: "admin",
        created_at: new Date("2025-03-26T00:00:00Z"),
        last_login: new Date("2025-03-26T12:00:00Z")
      },
      {
        username: "user1",
        password: "user123",
        role: "user",
        created_at: new Date("2025-03-26T00:00:00Z"),
        last_login: null
      }
    ]);
    console.log('👤 使用者資料新增完成');

    await users.deleteOne({ username: "user2" });
    console.log("🗑️ user2 刪除成功（如果存在的話）");

    // 2. Emergencies Collection
    const emergencies = db.collection("emergencies");
    await emergencies.insertMany([
      {
        emergency_id: "E001",
        elevator_id: "EVT001",
        reported_by: "U002",
        issue: "心肌梗塞",
        timestamp: new Date("2025-03-26T00:00:00Z"),
        status: "未解決"
      },
      {
        emergency_id: "E002",
        elevator_id: "EVT002",
        reported_by: "U001",
        issue: "卡住無法移動",
        timestamp: new Date("2025-03-26T00:00:00Z"),
        status: "已解決"
      }
    ]);
    console.log('🚨 緊急事件資料新增完成');

    // 3. Logs Collection
    const logs = db.collection("logs");
    await logs.insertMany([
      {
        log_id: "L001",
        emergency_id: "E001",
        action: "通知醫護人員",
        timestamp: new Date("2025-03-26T00:00:00Z"),
        note: "已派醫護人員到現場"
      },
      {
        log_id: "L002",
        emergency_id: "E002",
        action: "通知管理員",
        timestamp: new Date("2025-03-26T00:00:00Z"),
        note: "確認乘客"
      }
    ]);
    console.log('📋 處理紀錄資料新增完成');

    // 4. 更新 E001 狀態為「已解決」
    await emergencies.updateOne(
      { emergency_id: "E001" },
      { $set: { status: "已解決" } }
    );
    console.log("✅ 已將 E001 的狀態改為 '已解決'");

    const updatedE001 = await emergencies.findOne({ emergency_id: "E001" });
    console.log("🔍 查詢 E001：", updatedE001);

    // 5. Alert Notifications Collection
    const alertNotifications = db.collection("alert_notifications");
    await alertNotifications.insertMany([
      {
        alert_id: "A003",
        user_id: "user1",
        alert_message: "您的報告 E001 已經解決，感謝您的報告！",
        timestamp: new Date("2025-03-26T01:00:00Z")
      },
      {
        alert_id: "A004",
        user_id: "admin",
        alert_message: "電梯 EVT002 內有人昏倒，請派人進行急救。",
        timestamp: new Date("2025-03-26T01:30:00Z")
      }
    ]);
    console.log('📲 警報通知資料新增完成');

    // 6. Admin Dashboard Data Collection
    const dashboardData = db.collection("admin_dashboard_data");
    await dashboardData.insertMany([
      {
        report_date: new Date("2025-03-26T00:00:00Z"),
        total_emergencies: 5,
        total_resolved: 4,
        total_unresolved: 1,
        total_maintenance: 2
      },
      {
        report_date: new Date("2025-03-27T00:00:00Z"),
        total_emergencies: 3,
        total_resolved: 3,
        total_unresolved: 0,
        total_maintenance: 1
      }
    ]);
    console.log('📊 管理面板數據資料新增完成');

    // 7. System Settings Collection
    const systemSettings = db.collection("system_settings");
    await systemSettings.insertMany([
      {
        setting_name: "elevator_max_capacity",
        value: 12,
        description: "每台電梯的最大載客容量"
      },
      {
        setting_name: "emergency_alert_threshold",
        value: 5,
        description: "觸發緊急警報的最低事件數"
      }
    ]);
    console.log('⚙️ 系統設置資料新增完成');

    // 8. Elevators Collection
    const elevators = db.collection("elevators");
    await elevators.insertMany([
      {
        elevator_id: "EVT001",
        location: "大樓A - 東側電梯",
    
        capacity: 12,
        installed_date: new Date("2023-01-10")
      },
      {
        elevator_id: "EVT002",
        location: "大樓B - 西側電梯",
       
        capacity: 10,
        installed_date: new Date("2023-05-12")
      }
    ]);
    console.log("🏢 電梯資料新增完成");

    // 9. Elevator Operations Collection
   // 9. Elevator Operations Collection
   const elevatorOps = db.collection("elevator_operations");
   await elevatorOps.insertMany([
     {
       operation_id: "OP001",
       elevator_id: "EVT001",
       event: "正常運行",
       timestamp_start: new Date("2025-03-25T22:00:00Z"),
       timestamp_end: new Date("2025-03-26T04:00:00Z"),
       staff: "保全人員 小正",
       note: "每週例行巡查",
       status: 0 // ✅ 無狀況
     },
     {
       operation_id: "OP002",
       elevator_id: "EVT002",
       event: "處理緊急事件：E002",
       timestamp_start: new Date("2025-03-26T10:00:00Z"),
       timestamp_end: new Date("2025-03-26T11:00:00Z"),
       staff: "保全人員 阿呆",
       note: "電梯內乘客昏倒，已通報醫護",
       status: 1 // ✅ 有狀況
     }
   ]);
   console.log("📈 電梯運行紀錄新增完成");

    
    console.log("📈 電梯運行紀錄新增完成");

  } catch (err) {
    console.error('❌ 發生錯誤：', err);
  } finally {
    await client.close();
    console.log('🔚 已斷線');
  }
}

main();
