import telebot
import random
import string
import json
from keep_alive import keep_alive

keep_alive()

BOT_TOKEN = "8024432209:AAF9B1FWDswoGjnHnGKnKLiT4-zXSe6Buc4"
ADMIN_IDS = [6915752059]
bot = telebot.TeleBot(BOT_TOKEN)

history = []
profit = 0
user_turns = {}
DATA_FILE = "data.json"

def generate_nap_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def analyze_md5(md5_hash):
    global history

    algo1 = int(md5_hash[-2:], 16) % 2
    result1 = "Tài" if algo1 == 0 else "Xỉu"

    total_hex = sum(int(md5_hash[i:i+2], 16) for i in range(0, 8, 2))
    result2 = "Tài" if total_hex % 2 == 0 else "Xỉu"

    full_sum = sum(int(md5_hash[i:i+2], 16) for i in range(0, 32, 2))
    result3 = "Tài" if full_sum % 5 < 3 else "Xỉu"

    results = [result1, result2, result3]
    final_result = max(set(results), key=results.count)

    prediction = {
        "md5": md5_hash,
        "dự đoán": final_result,
        "thuật toán": {
            "thuật toán 1": result1,
            "thuật toán 2": result2,
            "thuật toán 3": result3,
        },
        "kết quả thực tế": None
    }
    history.append(prediction)

    return (f"✅ KẾT QUẢ PHÂN TÍCH PHIÊN TÀI XỈU MD5:\n"
            f"🔹 MD5: {md5_hash}\n\n"
            f"📊 Kết quả theo từng thuật toán:\n"
            f"   - Thuật toán 1 (2 ký tự cuối): {result1}\n"
            f"   - Thuật toán 2 (4 byte đầu): {result2}\n"
            f"   - Thuật toán 3 (Tổng toàn MD5): {result3}\n\n"
            f"✅ Kết luận cuối cùng: {final_result} | 🎯 Tín hiệu mạnh!\n"
            f"💡 Gợi ý: Cầu {final_result} đang lên mạnh!")

def save_data():
    with open(DATA_FILE, "w") as f:
        json.dump({"user_turns": user_turns, "history": history, "profit": profit}, f)

def load_data():
    global user_turns, history, profit
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            user_turns = data["user_turns"]
            history = data["history"]
            profit = data["profit"]
    except FileNotFoundError:
        save_data()

load_data()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "👋 Chào mừng đến với BOT TÀI XỈU VIP!\n"
                          "🔹 /tx <mã MD5> → Dự đoán kết quả (mỗi lần trừ 1 lượt).\n"
                          "🔹 /nap <số tiền> → Mua lượt dùng.\n"
                          "🔹 /dabank <số tiền> <nội dung> → Gửi thông tin giao dịch ngân hàng để admin duyệt.\n"
                          "🔹 /result <tài/xỉu> → Nhập kết quả thực tế (Admin).\n"
                          "🔹 /history → Xem lịch sử & lãi/lỗ.\n"
                          "🔹 /support → Liên hệ hỗ trợ.")

@bot.message_handler(commands=['tx'])
def get_tx_signal(message):
    user_id = message.from_user.id
    parts = message.text.split()
    if len(parts) < 2 or len(parts[1]) != 32:
        bot.reply_to(message, "❌ Vui lòng nhập mã MD5 hợp lệ!\n🔹 Ví dụ: /tx d41d8cd98f00b204e9800998ecf8427e")
        return

    turns = user_turns.get(user_id, 0)
    if turns <= 0:
        bot.reply_to(message, "⚠️ Bạn đã hết lượt dùng! Vui lòng dùng lệnh /nap <số tiền> để mua thêm.")
        return

    user_turns[user_id] = turns - 1
    save_data()
    md5_hash = parts[1]
    result_analysis = analyze_md5(md5_hash)
    bot.reply_to(message, result_analysis + f"\n\n🎫 Lượt còn lại: {user_turns[user_id]}")

@bot.message_handler(commands=['result'])
def set_actual_result(message):
    global profit
    if message.from_user.id not in ADMIN_IDS:
        bot.reply_to(message, "⛔ Bạn không có quyền sử dụng lệnh này!")
        return

    parts = message.text.split()
    if len(parts) < 2 or parts[1].lower() not in ["tài", "xỉu"]:
        bot.reply_to(message, "❌ Nhập kết quả hợp lệ! (tài/xỉu)")
        return

    actual_result = parts[1].capitalize()
    if not history:
        bot.reply_to(message, "⚠️ Chưa có dự đoán nào!")
        return

    last_prediction = history[-1]
    last_prediction["kết quả thực tế"] = actual_result

    if last_prediction["dự đoán"] == actual_result:
        profit += 1
        status = "✅ Thắng kèo! 📈 (+1 điểm)"
    else:
        profit -= 1
        status = "❌ Thua kèo! 📉 (-1 điểm)"

    save_data()
    bot.reply_to(message, f"📢 Cập nhật kết quả: {actual_result}\n{status}\n💰 Tổng lãi/lỗ: {profit} điểm")

@bot.message_handler(commands=['history'])
def show_history(message):
    if not history:
        bot.reply_to(message, "📭 Chưa có dữ liệu lịch sử!")
        return

    history_text = "📜 LỊCH SỬ DỰ ĐOÁN & KẾT QUẢ:\n"
    for idx, entry in enumerate(history[-5:], start=1):
        history_text += f"🔹 Lần {idx}:\n"
        history_text += f"   - 📊 Dự đoán: {entry['dự đoán']}\n"
        history_text += f"   - 🎯 Kết quả thực tế: {entry['kết quả thực tế'] or '❓ Chưa có'}\n"

    user_id = message.from_user.id
    turns = user_turns.get(user_id, 0)
    history_text += f"\n💰 Tổng lãi/lỗ: {profit} điểm\n🎫 Lượt còn lại: {turns}"
    bot.reply_to(message, history_text)

@bot.message_handler(commands=['nap'])
def handle_nap(message):
    parts = message.text.split()
    if len(parts) < 2 or not parts[1].isdigit():
        bot.reply_to(message, "❌ Vui lòng nhập số tiền hợp lệ! Ví dụ: /nap 100000")
        return

    amount = int(parts[1])
    user_id = message.from_user.id
    turns = amount // 1000
    if turns < 10 or turns > 10000:
        bot.reply_to(message, "⚠️ Bạn chỉ được mua từ 10 đến 10000 lượt (tương ứng từ 10,000đ đến 10,000,000đ).")
        return

    code = generate_nap_code()
    reply = (f"💳 HƯỚNG DẪN NẠP TIỀN MUA LƯỢT\n\n"
             f"➡️ Số tài khoản: 497720088\n"
             f"➡️ Ngân hàng: MB Bank\n"
             f"➡️ Số tiền: {amount} VNĐ\n"
             f"➡️ Nội dung chuyển khoản: NAP{code}\n\n"
             f"⏳ Sau khi chuyển khoản, admin sẽ duyệt và cộng {turns} lượt cho bạn.")

    for admin_id in ADMIN_IDS:
        bot.send_message(admin_id, f"📥 YÊU CẦU NẠP TIỀN\n"
                                   f"👤 User ID: {user_id}\n"
                                   f"💰 Số tiền: {amount} VNĐ\n"
                                   f"🎫 Lượt mua: {turns}\n"
                                   f"📝 Nội dung: NAP{code}\n\n"
                                   f"Duyệt bằng lệnh: /approve {user_id} {turns}")

    bot.reply_to(message, reply)

@bot.message_handler(commands=['approve'])
def approve_nap(message):
    if message.from_user.id not in ADMIN_IDS:
        return

    parts = message.text.split()
    if len(parts) < 3 or not parts[1].isdigit() or not parts[2].isdigit():
        bot.reply_to(message, "❌ Sai cú pháp. Dùng /approve <user_id> <số lượt>")
        return

    uid = int(parts[1])
    turns = int(parts[2])
    user_turns[uid] = user_turns.get(uid, 0) + turns

    save_data()
    bot.send_message(uid, f"✅ Bạn đã được cộng {turns} lượt dùng!\n🎯 Dùng lệnh /tx <md5> để dự đoán.")
    bot.reply_to(message, f"Đã cộng {turns} lượt cho user {uid}.")

@bot.message_handler(commands=['dabank'])
def handle_dabank(message):
    parts = message.text.split()
    if len(parts) < 3:
        bot.reply_to(message, "❌ Vui lòng nhập đầy đủ thông tin giao dịch. Ví dụ: /dabank 100000 Nội dung chuyển tiền")
        return

    amount = parts[1]
    content = " ".join(parts[2:])
    user_id = message.from_user.id

    for admin_id in ADMIN_IDS:
        bot.send_message(admin_id, f"📥 YÊU CẦU NẠP TIỀN (GIAO DỊCH NGÂN HÀNG)\n"
                                   f"👤 User ID: {user_id}\n"
                                   f"💰 Số tiền: {amount} VNĐ\n"
                                   f"📝 Nội dung: {content}\n\n"
                                   f"Duyệt bằng lệnh: /approve {user_id} <số lượt>")

    bot.reply_to(message, f"⏳ Đang chờ admin duyệt giao dịch.\n"
                          f"Sau khi admin duyệt, bạn sẽ nhận được lượt dùng.\n"
                          f"💰 Số tiền: {amount} VNĐ\n"
                          f"📝 Nội dung: {content}")

@bot.message_handler(commands=['support'])
def handle_support(message):
    bot.reply_to(message, "📩 Nếu bạn cần hỗ trợ, vui lòng liên hệ với admin tại: @cskhtool88")

bot.polling()