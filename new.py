    
from collections import deque
import numpy as np
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.text import Text
import shutil
from rich import box

class v1():
    class RoomPredictor(nn.Module):
        def __init__(self, num_rooms=8, embedding_dim=8, hidden_dim=32):
            super().__init__()
            self.embedding = nn.Embedding(num_rooms, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_rooms)

        def forward(self, x):
            x = self.embedding(x)
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    class RoomDataset(Dataset):
        def __init__(self, sequence, seq_len=5):
            self.inputs, self.targets = [], []
            for i in range(len(sequence) - seq_len):
                self.inputs.append(sequence[i:i + seq_len])
                self.targets.append(sequence[i + seq_len])

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

    # Train nhẹ (incremental)
    def train_model(model, history_rooms, epochs=3, seq_len=5):
        if len(history_rooms) <= seq_len:
            return
        dataset = v1.RoomDataset(history_rooms, seq_len)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for _ in range(epochs):
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    # Dự đoán bằng model
    def predict_with_model(model, last_seq):
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([last_seq], dtype=torch.long)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]
        return probs

    # Method 1: Tần suất recent100
    def method1_probs(recent100):
        counts_dict = recent100.get('room_id_2_killed_times', {})
        total = sum(counts_dict.values())
        probs = np.zeros(8)
        for r in range(1, 9):
            probs[r - 1] = counts_dict.get(str(r), 0) / total if total > 0 else 1/8
        return probs

    # Method 2: Tần suất recent10
    def method2_probs(recent10):
        rooms = [item['killed_room_id'] for item in recent10.get('data', [])]
        counts = np.bincount(rooms, minlength=9)[1:9]
        total = len(rooms)
        return counts/total if total > 0 else np.ones(8)/8

    # Method 3: Markov smoothing
    def method3_probs(history_rooms, alpha=1.0):
        if len(history_rooms) < 2:
            return np.ones(8)/8
        trans = np.zeros((8, 8)) + alpha
        for i in range(len(history_rooms) - 1):
            prev = history_rooms[i] - 1
            nxt = history_rooms[i + 1] - 1
            trans[prev, nxt] += 1
        last_room = history_rooms[-1] - 1
        row = trans[last_room]
        return row / row.sum()

    # === Main prediction ===
    def predict_safe_room(
        recent10, recent100, model, history,
        model_file=os.path.join("v2", "room_model.pth"), history_file=os.path.join("v2", "room_history.pkl"), last_result=None,
        auto_save=True
    ):
        # === Update history ===
        max_issue = max([h[0] for h in history], default=0)
        new_entries = [(item['issue_id'], item['killed_room_id']) for item in recent10.get('data', []) if item['issue_id'] > max_issue]
        if new_entries:
            history.extend(new_entries)
            history.sort(key=lambda x: x[0])
            if history_file is not None:
                with open(history_file, 'wb') as f:
                    pickle.dump(history, f)

        # === Chuẩn bị dữ liệu train ===
        history_rooms = [h[1] for h in history]
        history_rooms_0 = [r - 1 for r in history_rooms]

        # fine-tune nhẹ
        v1.train_model(model, history_rooms_0, epochs=3, seq_len=5)

        # === Auto-save mỗi N round ===
        if auto_save:
            if history_file is not None:
                with open(history_file, "wb") as f:
                    pickle.dump(history, f)
            torch.save(model.state_dict(), model_file)

        # === probs từ các phương pháp ===
        prob1 = v1.method1_probs(recent100)
        prob2 = v1.method2_probs(recent10)
        prob3 = v1.method3_probs(history_rooms)
        if len(history_rooms_0) >= 5:
            prob4 = v1.predict_with_model(model, history_rooms_0[-5:])
        else:
            prob4 = np.ones(8)/8

        # Ensemble weighted
        avg_prob = 0.35*prob2 + 0.3*prob1 + 0.2*prob3 + 0.15*prob4

        safe_scores = 1 - avg_prob
        # ===== Chọn phòng =====
        if last_result == "lose":
            top_each = [
                np.argmax(1 - prob1),
                np.argmax(1 - prob2),
                np.argmax(1 - prob3),
                np.argmax(1 - prob4),
            ]
            best_idx = np.argmax(safe_scores)
            votes = top_each.count(best_idx)
            if votes < 2:
                sorted_idx = np.argsort(safe_scores)[::-1]
                best_idx = sorted_idx[0] if safe_scores[sorted_idx[0]] - safe_scores[sorted_idx[1]] > 0.05 else sorted_idx[1]
            chosen_idx = best_idx
        else:
            top3_indices = np.argsort(safe_scores)[-3:]
            chosen_idx = random.choice(top3_indices)

        danger_scores = avg_prob
        scores = {i + 1: float(danger_scores[i]) for i in range(8)}

        return chosen_idx + 1, history
class v2():
    class DQN(nn.Module):
        def __init__(self, state_dim=16, action_dim=8, hidden_dim=128):
            super(v2.DQN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        def forward(self, x):
            return self.net(x)


    class DQNAgent:
        def __init__(self, state_dim=16, action_dim=8, gamma=0.95, lr=0.001,
                    epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                    buffer_size=5000, batch_size=64, model_file=os.path.join("v2", "dqn_model.pth")):

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.memory = deque(maxlen=buffer_size)
            self.model_file = model_file

            self.model = v2.DQN(state_dim, action_dim)
            self.target_model = v2.DQN(state_dim, action_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.criterion = nn.MSELoss()

            # Load model nếu có sẵn
            if os.path.exists(model_file):
                self.model.load_state_dict(torch.load(model_file))
                self.target_model.load_state_dict(torch.load(model_file))
            else:
                self.update_target()
        def build_state(recent10, recent100):
            freq100 = np.array(list(recent100["data"]["room_id_2_killed_times"].values()))
            rooms10 = [item['killed_room_id'] for item in recent10['data']]
            freq10 = np.bincount(rooms10, minlength=9)[1:9]
            state = np.concatenate([freq100/100.0, freq10/10.0])
            return state
        def update_target(self):
            self.target_model.load_state_dict(self.model.state_dict())

        def save(self):
            torch.save(self.model.state_dict(), self.model_file)

        def act_full(self, recent10, recent100, reward=None):
            """
            recent10, recent100: dữ liệu raw từ API
            reward: nếu có thì agent sẽ học luôn, nếu None thì chỉ chọn hành động
            """
            # build state
            state = v2.DQNAgent.build_state(recent10, recent100)

            # chọn action
            if np.random.rand() < self.epsilon:
                action = random.randrange(self.action_dim)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state_tensor).squeeze(0)
                action = torch.argmax(q_values).item()

            # Tính score nguy hiểm
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
            scores = {i + 1: float(-q_values[i].item()) for i in range(self.action_dim)}

            # Nếu có reward thì update luôn
            if reward is not None and hasattr(self, "last_state") and hasattr(self, "last_action"):
                self.remember(self.last_state, self.last_action, reward, state)
                self.replay()

            # Lưu state + action hiện tại để lần sau học
            self.last_state = state
            self.last_action = action

            return action   # room_id từ 1–8



        def remember(self, state, action, reward, next_state):
            self.memory.append((state, action, reward, next_state))

        def replay(self):
            if len(self.memory) < self.batch_size:
                return
            minibatch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states = zip(*minibatch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)

            q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values

            loss = self.criterion(q_values, targets.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
import requests
import hashlib, hmac, time, uuid, random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import requests
from colorama import Fore, Style
from urllib.parse import urlparse, parse_qs
import json
class Link():
    def tao_link4m(long_url):
        api_token = "67b869a78c74e043197209f3"  # thay bằng token thật của bạn
        api_url = "https://link4m.co/api-shorten/v2"
        params = {
            "api": api_token,
            "url": long_url
        }
        resp = requests.get(api_url, params=params)
        data = resp.json()

        if data.get("status") == "success":
            return data["shortenedUrl"]
        else:
            return f"Lỗi: {data.get('message')}"

# ===== DÙNG LUÔN Ở ĐÂY =====
    def tao_link(key):
        url = "https://anotepad.com/note/create"
        files = {
            "notetype": (None, "PlainText"),
            "noteaccess": (None, "2"),
            "notecontent": (None, key),
        }
        res = requests.post(url, files=files)
        try:
            data = res.json()
            return "https://anotepad.com/notes/" + data["notenumber"]
        except Exception as e:
            print("Lỗi:", e, res.text)
        
class Key():
    def __init__(self):
        self.SECRET_KEY = b"dit-me-may-hack-cai-lon-bypass-cai-lon-du-me-cho-sai-free-con-bypass-u-ma-may-Lionleo@1237"
        self.ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def base36_encode(self,num: int, length: int) -> str:
        s = ""
        while num > 0:
            num, r = divmod(num, 36)
            s = self.ALPHABET[r] + s
        return s.rjust(length, "A")[:length]

    def base36_decode(self,s: str) -> int:
        num = 0
        for c in s:
            num = num * 36 + self.ALPHABET.index(c)
        return num

    def get_device_code(self) -> str:
        hwid = str(uuid.getnode())
        h = hashlib.sha256(hwid.encode()).hexdigest()
        return h[:8]

    def create_key(self,device_code: str, hours: int, typ: str) -> str:
        exp = int(time.time()) + hours * 3600
        exp_code = self.base36_encode(exp // 60, 6)
        dev_code = self.base36_encode(int(device_code[:6], 16), 3)
        typ_char = "V" if typ == "vip" else "F"
        nonce = self.base36_encode(random.randint(0, 36**2 - 1), 2)
        data = exp_code + dev_code + typ_char + nonce
        sig = hmac.new(self.SECRET_KEY, data.encode(), hashlib.sha256).hexdigest()
        chk = self.base36_encode(int(sig, 16), 2)
        return "Czero" + data + chk

    def check_key(self,key: str, device_code: str):
        if not key.startswith("Czero") or len(key) != 19:
            return False, "Sai định dạng key"

        body = key[5:]
        exp_code, dev_code, typ_char, nonce, chk = body[:6], body[6:9], body[9], body[10:12], body[12:]
        exp = self.base36_decode(exp_code) * 60

        # check expiry
        if time.time() > exp:
            return False, "❌ Key đã hết hạn"

        # check device
        dev_check = self.base36_encode(int(device_code[:6], 16), 3)
        if dev_code != dev_check:
            return False, "❌ Key không dành cho thiết bị này"

        # verify signature
        data = exp_code + dev_code + typ_char + nonce
        sig = hmac.new(self.SECRET_KEY, data.encode(), hashlib.sha256).hexdigest()
        good_chk = self.base36_encode(int(sig, 16), 2)
        if chk != good_chk:
            return False, "❌ Key không hợp lệ (bị sửa)"

        typ = "VIP" if typ_char == "V" else "FREE"
        return True, f"✅ Key hợp lệ ({typ}), hết hạn: {time.ctime(exp)}"

# ===== DÙNG LUÔN Ở ĐÂY =====


# ===== DÙNG LUÔN Ở ĐÂY =====
import os

def dua_link_key():
    key_manager = Key()
    device = key_manager.get_device_code()
    print(f"Device:{device}")

    # Quy trình lấy key:
    print("=== Kiểm tra key ===")
    chon_key=input("Bạn có muốn đổi key (y/n)? ").strip().lower()
    if chon_key in ["y","yes"]:
        if os.path.exists("key.txt"):
            os.remove("key.txt")
            print("Đã xóa key cũ, sẽ tạo key mới...")
        else:
            print("Chưa có key cũ, sẽ tạo key mới...")
    elif chon_key in ["n","no",""]:
        print("Sẽ dùng key cũ.")
    # 1. kiểm tra file key.txt
        if os.path.exists("key.txt"):
            with open("key.txt", "r") as f:
                saved_key = f.read().strip()
            ok, msg = key_manager.check_key(saved_key, device)
            print(msg)
            if ok:
                return saved_key  # key hợp lệ thì trả luôn

    # 2. nếu chưa có file hoặc key sai/hết hạn -> tạo mới
    new_key = key_manager.create_key(device, 24, "free")
    print(new_key)

    # 3. tạo link anotepad chứa key
    note_link = Link.tao_link(new_key)
    short_link = Link.tao_link4m(note_link)
    print("👉 Vào link này để lấy key:", short_link)

    # 4. yêu cầu nhập key đến khi đúng
    while True:
        user_key = input("Nhập key của bạn: ").strip()
        ok, msg = key_manager.check_key(user_key, device)
        print(msg)
        if ok:
            with open("key.txt", "w") as f:
                f.write(new_key)
            return user_key
        else:
            print("Key sai hoặc hết hạn, thử lại...")
class HienThi():
    def __init__(self):
        self.console = Console()

    def show_stats(self,round_idx, win_count, lose_count, acc, max_chuoi_thang, max_lose_streak, loi):
        table = Table(title=f"📊 KẾT QUẢ SAU {round_idx} VÁN", box=box.ROUNDED, expand=True)

        table.add_column("📈 Thắng", justify="center", style="green", no_wrap=True)
        table.add_column("📉 Thua", justify="center", style="red", no_wrap=True)
        table.add_column("🎯 Chính xác", justify="center", style="cyan", no_wrap=True)
        table.add_column("🔥 Chuỗi thắng max", justify="center", style="yellow", no_wrap=True)
        table.add_column("💀 Chuỗi thua max", justify="center", style="magenta", no_wrap=True)
        table.add_column("💰 Lời/Lỗ", justify="center", style="bold", no_wrap=True)

        table.add_row(
            str(win_count),
            str(lose_count),
            f"{acc:.2f}%",
            str(max_chuoi_thang),
            str(max_lose_streak),
            str(round(loi,2))
        )

        self.console.print(Panel(table, title="🔥 Thống kê", border_style="bright_blue"))


    def show_predict(self,issue_id, best_room, dict_room):
        self.console.print(Panel.fit(
            f"🔮 Dự đoán ẩn phòng [bold yellow]{best_room}[/] "
            f"({dict_room.get(str(best_room), '???')}) "
            f"cho phiên [cyan]{issue_id}[/]",
            border_style="green"
        ))


    def show_reward(self,loi):
        # chọn màu chữ dựa vào giá trị
        if loi > 0:
            color = "green"
        elif loi < 0:
            color = "red"
        else:
            color = "white"

        # tạo text với 2 phần khác màu
        text = Text()
        text.append("Lời/lỗ: ", style="yellow")
        text.append(str(loi), style=color)

        # gói trong Panel cho đẹp
        panel = Panel.fit(
            text,
            border_style="bright_magenta",   # gợi ý: viền tím nổi bật
            padding=(0,1)                    # thêm khoảng cách cho gọn
        )

        self.console.print(panel)

    def show_result(self,killed, best_room, dict_room, win):
        if win:
            self.console.print(f"✔ [green]ĐÚNG[/] — Killer vào [red]{dict_room[str(killed)]}[/], bạn ẩn ở [yellow]{dict_room[str(best_room)]}[/]")
        else:
            self.console.print(f"✘ [red]SAI[/] — Killer vào [red]{dict_room[str(killed)]}[/], bạn ẩn ở [yellow]{dict_room[str(best_room)]}[/]")


    def show_summary(self,loi, chuoi_thang_hien_tai, chuoi_thang_max, chuoi_thua_dict):
        # --- Lời/Lỗ ---
        if loi > 0:
            color = "green"
        elif loi < 0:
            color = "red"
        else:
            color = "white"

        line1 = Text()
        line1.append("Lời/lỗ: ", style="yellow")
        line1.append(str(loi), style=color)

        # --- Chuỗi thắng ---
        line2 = Text()
        line2.append("Chuỗi thắng: ", style="yellow")
        line2.append(str(chuoi_thang_hien_tai), style="green")
        line2.append(f" (max: {chuoi_thang_max})", style="cyan")

        # --- Chuỗi thua ---
        line3 = Text()
        line3.append("Chuỗi thua: ", style="yellow")
        parts = [f"{key}/({val})" for key, val in chuoi_thua_dict.items()]
        line3.append(" | ".join(parts), style="red")

        # Gộp text vào panel
        text = Text()
        text.append(line1)
        text.append("\n")
        text.append(line2)
        text.append("\n")
        text.append(line3)

        panel = Panel.fit(
            text,
            border_style="bright_magenta",  # viền đẹp, nổi bật
            padding=(0, 1)
        )
        self.console.print(panel)

    def show_results(self,win_count, lose_count, acc):
        total = win_count + lose_count

        # tạo text nhiều dòng
        text = Text()
        text.append(f"🎮 Tổng ván: ", style="cyan")
        text.append(f"{total}\n", style="bold white")

        text.append(f"✅ Ván thắng: ", style="green")
        text.append(f"{win_count}\n", style="bold white")

        text.append(f"❌ Ván thua: ", style="red")
        text.append(f"{lose_count}\n", style="bold white")

        text.append(f"🎯 Tỉ lệ thắng: ", style="yellow")
        text.append(f"{acc:.2f}%", style="bold white")

        # panel chứa text
        panel = Panel.fit(
            text,
            title="📈 Kết quả",
            border_style="magenta",
            padding=(0, 1),
        )
        self.console.print(panel)
    def show_rest(self,van_nghi_con_lai: int):
        
        self.console.print(Panel.fit(
            f"😴 Đang nghỉ {van_nghi_con_lai} ván..."
            ,
            border_style="green"
        ))

        
#=====API=====
def get_recent_100(HEADERS,PARAMS):
    try:
        url = 'https://api.escapemaster.net/escape_game/recent_100_issues'
        response = requests.get(url, headers=HEADERS, params=PARAMS)
        return response.json()
    except Exception as e:
        print("Lỗi gọi API recent_100_issues:", e)
        return False

def get_recent_10(HEADERS,PARAMS):
    try:
        url = 'https://api.escapemaster.net/escape_game/recent_10_issues'
        response = requests.get(url, headers=HEADERS, params=PARAMS)
        return response.json()
    except Exception as e:
        print("Lỗi gọi API recent_10_issues:", e)
        return False
def lay_ket_qua(HEADERS,type):
    try:
        params = {
            'asset': f'{type}',
            'page': '1',
            'page_size': '10',
        }
        response = requests.get('https://api.escapemaster.net/escape_game/my_joined', params=params, headers=HEADERS).json()
        return response
    except Exception as e:
        print("Lỗi gọi API my_joined:", e)
        return False
class ket_qua:
    def place_bet_simple(room_id: int,bet,user_id,secret_id,bet_type):
        headers_ = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': 'https://escapemaster.net',
            'priority': 'u=1, i',
            'referer': 'https://escapemaster.net/',
            'sec-ch-ua': '"Not;A=Brand";v="99", "Microsoft Edge";v="139", "Chromium";v="139"',
            'sec-ch-ua-mobile': '?1',
            'sec-ch-ua-platform': '"Android"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; SM-G955U Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36 Edg/139.0.0.0',
            'user-id': user_id,
            'user-login': 'login_v2',
            'user-secret-key': secret_id,
        }

        json_data = {
            'asset_type': f'{bet_type}',
            'user_id': user_id,
            'room_id': room_id,
            'bet_amount': bet,
        }
        try:
            response = requests.post('https://api.escapemaster.net/escape_game/bet', headers=headers_, json=json_data).json()
            return response
        except Exception as e:
            print("Lỗi gọi API đặt cược:", e)
            return False

CONFIG_FILE = "config.json"
def print_divider(char="="):
    width = shutil.get_terminal_size().columns
    print(char * width)
def load_or_create_config():
    if os.path.exists(CONFIG_FILE):
        choice = input("\n 🔎 Đã lưu config, bạn có muốn dùng lại không? (y/n): ").strip().lower()
        if choice in ["y","yes",""]:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        if choice in ["n","no"]:
            print("♻️ Nhập lại config mới:")
    else:
        print("⚠️ Chưa có config, hãy nhập mới:")

    link_game = input(Fore.CYAN + "Nhập Link Game: " + Style.RESET_ALL).strip()
    while True:
        bet_type_input = input("Nhập Loại Tiền cược BUILD/USDT/WORLD (1/2/3): ").strip()
        if bet_type_input == "1":
            bet_type = "BUILD"
            break
        elif bet_type_input == "2":
            bet_type = "USDT"
            break
        elif bet_type_input == "3":
            bet_type = "WORLD"
            break
        else:
            print("Loại tiền không hợp lệ, hãy nhập lại.")
    config = {"url_game": link_game, "bet_type": bet_type}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(Fore.GREEN + f"✅ Đã lưu config vào {CONFIG_FILE}")
    return config
def parse_range_input(text):
    """Xử lý input dạng '4' hoặc '4,6' -> trả về số random trong khoảng"""
    try:
        parts = text.split(",")
        if len(parts) == 1:
            return int(parts[0])
        elif len(parts) == 2:
            return random.randint(int(parts[0]), int(parts[1]))
    except:
        print("Input sai định dạng, dùng mặc định 4")
        pass
    return 4

# ================== LOOP CHÍNH ==================
def main_loop():
    key=dua_link_key()
    config = load_or_create_config()
    parsed_url = urlparse(config["url_game"]) 
    params = parse_qs(parsed_url.query)
    user_id = params.get("userId", [None])[0]
    secret_key = params.get("secretKey", [None])[0]
    bet_type = config["bet_type"]
    bet_amoun=float(input("Nhập số build cược:"))
    he_so=float(input("Hệ số sau thua:"))
    so_van_thang_in = input("Nghỉ sau bao nhiêu ván thắng liên tiếp (vd: 3 hoặc 4,6): ").strip()
    so_van_nghi_in = input("Số ván nghỉ (vd: 2 hoặc 1,3): ").strip()

    so_van_thang_target = parse_range_input(so_van_thang_in)
    so_van_nghi_target = parse_range_input(so_van_nghi_in)
    van_nghi_con_lai = 0
    key_manager = Key()
    device = key_manager.get_device_code()
    ok, msg = key_manager.check_key(key, device)
    print(msg)
    if not ok:
        return
    if "(VIP)" in msg:
        mod = "AI_v1"
    else:
        mod = "AI_v2"
    last_van=None
    HEADERS = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'country-code': 'vn',
    'origin': 'https://xworld.info',
    'priority': 'u=1, i',
    'referer': 'https://xworld.info/',
    'sec-ch-ua': '"Not;A=Brand";v="99", "Microsoft Edge";v="139", "Chromium";v="139"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'cross-site',
    'user-id': user_id,
    'user-login': 'login_v2',
    'user-secret-key': secret_key,
    'xb-language': 'vi-VN',
}
    PARAMS = {'asset': f'{bet_type}'}
    last_bet=bet_amoun
    DICT_ROOM = {
        "1": "Nhà Kho",
        "2": "Phòng Họp",
        "3": "Phòng Giám Đốc",
        "4": "Phòng Trò Chuyện",
        "5": "Phòng Giám Sát",
        "6": "Văn Phòng",
        "7": "Phòng Tài Vụ",
        "8": "Phòng Nhân Sự"
    }
    round_idx = 0
    win_count = 0
    lose_count = 0
    lose_streak = 0
    max_lose_streak = 0
    chuoi_thang=0
    max_chuoi_thang=0
    chuoi_thang_nghi=0
    ds_chuoi_thua={}
    loi=0
    dat=False
    r=None
    ht = HienThi()
    while True:
        round_idx += 1

        # Lấy dữ liệu API
        # try:
        recent10 = get_recent_10(HEADERS,PARAMS)
        recent100 = get_recent_100(HEADERS,PARAMS)
        if not recent10 or not recent100 or recent10['code'] != 0 or recent100['code'] != 0:
            print(Fore.RED + "Lỗi gọi API lịch sử, thử lại sau..." + Style.RESET_ALL)
            time.sleep(2)
            continue
        # except Exception as e:
        #     print(Fore.RED + f"Lỗi gọi API lịch sử: {e}" + Style.RESET_ALL)
        #     time.sleep(3)
        #     continue

        try:
            current_issue_id = str(int(recent10['data'][0]["issue_id"])+1)
        except Exception:
            time.sleep(3)
            continue
        
        # Load history
        
        if van_nghi_con_lai > 0:
            ht.show_rest(van_nghi_con_lai)
            # print(Fore.YELLOW + f"😴 Đang nghỉ {van_nghi_con_lai} ván..." + Style.RESET_ALL)
            van_nghi_con_lai -= 1
            nghi=True
        else:
            nghi=False
        # ====== GỌI predict_safe_room ======
        # try:
        #     best_room_v1, history = v1.predict_safe_room(
        #         recent10, recent100, model, history,
        #         model_file= "room_model.pth",
        #         history_file="room_history.pkl",
        #         last_result=last_van
        #     )
        # except Exception as e:
        #     print(Fore.RED + f"Lỗi khi gọi predict_safe_room: {e}" + Style.RESET_ALL)
        #     time.sleep(3)
        #     continue

        # print(f"Lời-lỗ: {loi}")
        if not nghi:
            if mod=="AI_v1":
                history_file =  "room_history.pkl"
                model_file = "room_model.pth"
                if os.path.exists(history_file):
                    with open(history_file, "rb") as f:
                        history = pickle.load(f)
                else:
                    history = []

                # Load model
                model = v1.RoomPredictor()
                if os.path.exists(model_file):
                    model.load_state_dict(torch.load(model_file))
                try:
                    best_room_v1, history = v1.predict_safe_room(
                        recent10, recent100, model, history,
                        model_file= "room_model.pth",
                        history_file="room_history.pkl",
                        last_result=last_van
                )
                except Exception as e:
                    print(Fore.RED + f"Lỗi khi gọi predict_safe_room: {e}" + Style.RESET_ALL)
                    time.sleep(3)
                    continue
                # print(f"🔮Dự đoán ẩn phòng [{best_room_v1}] cho phiên {current_issue_id}")
                ht.show_predict(str(current_issue_id), best_room_v1, DICT_ROOM)
                best_room=best_room_v1
                dat=True
            elif mod=="AI_v2":
                agent=v2.DQNAgent()
                action=agent.act_full(recent10=recent10,recent100=recent100,reward=r)
                best_room_v2=action+1
                # print(f"🔮Dự đoán ẩn phòng [{best_room_v2}] cho phiên {current_issue_id}")
                ht.show_predict(str(current_issue_id), best_room_v2, DICT_ROOM)
                best_room=best_room_v2
                dat=True
            elif mod=="AI_v12":
                if best_room_v1==best_room_v2:
                    print("Cuoc theo AI_v12")
                    best_room=best_room_v1
                    dat=True
        else:
            dat=False
        # ht.show_reward(round(loi,4))
        ht.show_summary(round(loi,4), chuoi_thang, max_chuoi_thang, ds_chuoi_thua)
        if dat:
            datcuoc=ket_qua.place_bet_simple(int(best_room),last_bet,user_id,secret_key,bet_type)
            if datcuoc['msg']=='ok':
                print(f'Đã cược {last_bet} vào phòng {DICT_ROOM[str(best_room)]}')
            else:
                print(f'Lỗi đặt cược!\n{datcuoc}')
        
        # ====== Lấy kết quả thực tế ======
        WAIT_TIMEOUT = 80
        CHECK_EVERY = 2
        elapsed = 0
        result = None
        
        while elapsed < WAIT_TIMEOUT:
            time.sleep(CHECK_EVERY)
            elapsed += CHECK_EVERY
            try:
                now10 = get_recent_10(HEADERS,PARAMS)
                if not now10 or now10['code'] != 0:
                    continue

                latest_issue_id = str(now10['data'][0]["issue_id"])
                if int(latest_issue_id) >= int(current_issue_id):
                    print(f"Phiên [{current_issue_id}] đã kết thúc.")
                    result = now10['data'][0]
                    break
                else:
                    print(f"⏳ Chờ... {elapsed}s", end="\r", flush=True)

            except Exception as e:
                print("Lỗi check my_joined:", e)

        if result is None:
            print("\n⏱️ Hết thời gian chờ, bỏ qua lần này.\n" + "="*90 + "\n")
            continue

        # ====== ĐÁNH GIÁ ======
        try:
            killed = int(result["killed_room_id"])
            # reward=round(lay_ket_qua(HEADERS)['data']['items'][0]["award_amount"],4)-last_bet
            # loi+=reward
        except Exception as e:
            print(Fore.RED + f"Không đọc được kết quả ván: {e}" + Style.RESET_ALL)
            continue
        if dat:
            if killed != best_room:
                last_van='win'
                win_count += 1
                lose_streak = 0
                chuoi_thang+=1
                reward=round(lay_ket_qua(HEADERS,type=bet_type)['data']['items'][0]["award_amount"],4)-last_bet
                loi+=reward
                last_bet=bet_amoun
                r=1
                max_chuoi_thang=max(max_chuoi_thang,chuoi_thang)
                # 1
                #print(Fore.GREEN + f"✔ ĐÚNG — Killer vào {DICT_ROOM[str(killed)]}, bạn ẩn ở {DICT_ROOM[str(best_room)]}" + Style.RESET_ALL)
                ht.show_result(killed, best_room, DICT_ROOM, win=True)
                chuoi_thang_nghi += 1
                if chuoi_thang_nghi >= so_van_thang_target:
                    print(Fore.CYAN + f"🎉 Đã thắng liên tiếp {chuoi_thang_nghi} ván, nghỉ {so_van_nghi_target} ván!" + Style.RESET_ALL)
                    van_nghi_con_lai = so_van_nghi_target
                    # reset chuỗi thắng, random lại target
                    chuoi_thang_nghi = 0
                    so_van_thang_target = parse_range_input(so_van_thang_in)
                    so_van_nghi_target = parse_range_input(so_van_nghi_in)
            else:
                chuoi_thang_nghi = 0
                last_van='lose'
                lose_count += 1
                lose_streak += 1
                chuoi_thang=0
                ds_chuoi_thua[f"{lose_streak}"] = ds_chuoi_thua.get(f"{lose_streak}", 0) + 1
                if dat:
                    last_bet=last_bet*he_so
                r=-1
                max_lose_streak = max(max_lose_streak, lose_streak)
                # print(Fore.RED + f"✘ SAI — Killer vào {DICT_ROOM[str(killed)]}, bạn ẩn ở {DICT_ROOM[str(best_room)]}" + Style.RESET_ALL)
                ht.show_result(killed, best_room, DICT_ROOM, win=False)
        
        total = win_count + lose_count
        acc = 100.0 * win_count / total if total > 0 else 0.0
        ht.show_results(win_count, lose_count, acc)

        # print(f"📊 Tổng ván: {total} | Thắng: {win_count} | Thua: {lose_count}")
        
        # print(f"🎯 Độ chính xác: {acc:.2f}% | Chuỗi thắng cao nhất: {max_chuoi_thang} | Chuỗi thua dài nhất: {max_lose_streak}")
        # print("=" * 90)
        print_divider("=")


if __name__ == "__main__":
    main_loop()

