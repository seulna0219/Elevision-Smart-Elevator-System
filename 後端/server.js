const express = require('express');
const cors = require('cors');
const { MongoClient, ObjectId } = require('mongodb');

const app = express();
const PORT = 3000;

// MongoDB 連線設定
const uri = 'mongodb://127.0.0.1:27017';
const client = new MongoClient(uri);
let db;

// 中介軟體：處理跨域與 JSON 請求
app.use(cors());
app.use(express.json());

// ✅ 連接資料庫
async function connectDB() {
  try {
    await client.connect();
    db = client.db('elevator_emergency_dbb');
    console.log('✅ 已連接資料庫');
  } catch (error) {
    console.error('❌ 資料庫連線失敗', error);
  }
}
connectDB();

// ✅ 查詢所有緊急事件（GET）
app.get('/api/emergencies', async (req, res) => {
  try {
    const emergencies = await db.collection('emergencies').find().toArray();
    res.json(emergencies);
  } catch (error) {
    res.status(500).json({ error: '❌ 查詢失敗' });
  }
});

// ✅ 新增一筆緊急事件（POST）
app.post('/api/emergencies', async (req, res) => {
  try {
    const newEmergency = req.body;
    const result = await db.collection('emergencies').insertOne(newEmergency);
    res.json({ message: '✅ 新增成功', id: result.insertedId });
  } catch (error) {
    res.status(500).json({ error: '❌ 新增失敗' });
  }
});

// ✅ 刪除一筆緊急事件（DELETE）
app.delete('/api/emergencies/:id', async (req, res) => {
  try {
    const id = req.params.id;
    await db.collection('emergencies').deleteOne({ _id: new ObjectId(id) });
    res.json({ message: '✅ 刪除成功' });
  } catch (error) {
    res.status(500).json({ error: '❌ 刪除失敗' });
  }
});

// ✅ 更新一筆緊急事件（PUT）
app.put('/api/emergencies/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const updateData = req.body;
    await db.collection('emergencies').updateOne(
      { _id: new ObjectId(id) },
      { $set: updateData }
    );
    res.json({ message: '✅ 更新成功' });
  } catch (error) {
    res.status(500).json({ error: '❌ 更新失敗' });
  }
});

// ✅ 啟動伺服器
app.listen(PORT, () => {
  console.log(`🚀 API 伺服器運行中：http://localhost:${PORT}`);
});
