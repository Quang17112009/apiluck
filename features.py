import pandas as pd
from typing import List

# Định nghĩa thứ tự và tên của các cột tính năng.
# RẤT QUAN TRỌNG: Phải khớp với thứ tự và tên cột mà mô hình được huấn luyện.
# Các tính năng được mở rộng.
FEATURE_COLUMNS = [
    'last_outcome_is_tai',
    'length_of_current_streak',
    'tai_ratio_last_5',
    'xiu_ratio_last_5',
    'tai_ratio_last_10',
    'xiu_ratio_last_10',
    'tai_ratio_last_20',
    'xiu_ratio_last_20',
    'num_switches_last_5',
    'num_switches_last_10',
    'is_alternating_last_4', # T-X-T-X hoặc X-T-X-T
    'is_two_streak_alternating_last_6', # TT-XX-TT hoặc XX-TT-XX
    'longest_tai_streak_last_20',
    'longest_xiu_streak_last_20',
]

def _calculate_streak(results: List[str], outcome_type: str) -> int:
    """Tính độ dài chuỗi liên tiếp của một loại kết quả từ đầu danh sách."""
    count = 0
    for res in results:
        if res == outcome_type:
            count += 1
        else:
            break
    return count

def _get_longest_streak(results: List[str], outcome_type: str) -> int:
    """Tìm độ dài chuỗi dài nhất của một loại kết quả trong danh sách."""
    if not results:
        return 0
    max_streak = 0
    current_streak = 0
    for i in range(len(results)):
        if results[i] == outcome_type:
            current_streak += 1
        else:
            max_streak = max(max_streak, current_streak)
            current_streak = 0
    return max(max_streak, current_streak) # Bao gồm cả chuỗi cuối cùng

def extract_features(historical_results_list: List[str]) -> pd.DataFrame:
    """
    Trích xuất các đặc trưng từ danh sách kết quả lịch sử.
    historical_results_list: Danh sách các chuỗi kết quả ('Tài' hoặc 'Xỉu'),
                             được sắp xếp từ MỚI NHẤT đến CŨ NHẤT.
    Trả về một pd.DataFrame với các tính năng.
    """
    features = {}

    # Nếu không có lịch sử, trả về DataFrame với giá trị 0 mặc định cho tất cả cột
    if not historical_results_list:
        return pd.DataFrame([[0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)

    # Lấy kết quả gần nhất
    last_outcome = historical_results_list[0]
    features['last_outcome_is_tai'] = 1 if last_outcome == 'Tài' else 0

    # Độ dài cầu hiện tại (cầu của kết quả gần nhất)
    features['length_of_current_streak'] = _calculate_streak(historical_results_list, last_outcome)

    # Tỷ lệ Tài/Xỉu trong các cửa sổ khác nhau
    for N in [5, 10, 20]:
        recent_results = historical_results_list[:N]
        tai_count = recent_results.count("Tài")
        xiu_count = recent_results.count("Xỉu")
        total_count = len(recent_results)

        features[f'tai_ratio_last_{N}'] = tai_count / total_count if total_count > 0 else 0
        features[f'xiu_ratio_last_{N}'] = xiu_count / total_count if total_count > 0 else 0

        # Số lần chuyển đổi (switch) trong N phiên gần nhất
        num_switches = 0
        if total_count >= 2:
            for i in range(total_count - 1):
                if recent_results[i] != recent_results[i+1]:
                    num_switches += 1
        if N == 5 or N == 10: # Chỉ tính cho N=5 và N=10 theo FEATURE_COLUMNS
            features[f'num_switches_last_{N}'] = num_switches

    # Mẫu cầu đảo (alternating pattern)
    # T-X-T-X hoặc X-T-X-T trong 4 phiên gần nhất
    is_alternating_last_4 = False
    if len(historical_results_list) >= 4:
        sub_list = historical_results_list[:4]
        if (sub_list[0] != sub_list[1] and 
            sub_list[1] != sub_list[2] and 
            sub_list[2] != sub_list[3]):
            is_alternating_last_4 = True
    features['is_alternating_last_4'] = 1 if is_alternating_last_4 else 0

    # Mẫu cầu đảo kép (two-streak alternating pattern)
    # TT-XX-TT hoặc XX-TT-XX trong 6 phiên gần nhất
    is_two_streak_alternating_last_6 = False
    if len(historical_results_list) >= 6:
        sub_list = historical_results_list[:6]
        if (sub_list[0] == sub_list[1] and sub_list[1] != sub_list[2] and
            sub_list[2] == sub_list[3] and sub_list[3] != sub_list[4] and
            sub_list[4] == sub_list[5]):
            is_two_streak_alternating_last_6 = True
    features['is_two_streak_alternating_last_6'] = 1 if is_two_streak_alternating_last_6 else 0

    # Độ dài chuỗi Tài và Xỉu dài nhất trong 20 phiên gần nhất
    recent_20_results = historical_results_list[:20]
    features['longest_tai_streak_last_20'] = _get_longest_streak(recent_20_results, 'Tài')
    features['longest_xiu_streak_last_20'] = _get_longest_streak(recent_20_results, 'Xỉu')

    # Chuyển đổi thành DataFrame và đảm bảo thứ tự cột KHỚP với lúc huấn luyện
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def create_training_data(all_historical_results_strings: List[str]) -> (pd.DataFrame, pd.Series):
    """
    Tạo tập dữ liệu huấn luyện (features X và labels y) từ tất cả các kết quả lịch sử.
    all_historical_results_strings: Danh sách các chuỗi kết quả ('Tài' hoặc 'Xỉu'),
                                    được sắp xếp từ MỚI NHẤT đến CŨ NHẤT.
    """
    features_list = []
    labels = []

    # Chúng ta muốn dự đoán kết quả của phiên tại index `label_idx`
    # dựa trên lịch sử các phiên `historical_results_strings[label_idx + 1:]`
    
    # Cần ít nhất 2 phiên để tạo một cặp (lịch sử, nhãn)
    if len(all_historical_results_strings) < 2:
        return pd.DataFrame(), pd.Series(dtype=str)

    # Lặp từ index 0 đến len - 2 để lấy nhãn và lịch sử tương ứng
    for label_idx in range(len(all_historical_results_strings) - 1):
        label = all_historical_results_strings[label_idx]
        history_for_features = all_historical_results_strings[label_idx + 1:] # Lịch sử là các phiên cũ hơn nhãn

        # Chỉ tạo features nếu có đủ lịch sử để extract_features không trả về rỗng
        if history_for_features:
            features_row_df = extract_features(history_for_features)
            # Kiểm tra nếu features_row_df không rỗng và có đúng các cột
            if not features_row_df.empty and list(features_row_df.columns) == FEATURE_COLUMNS:
                features_list.append(features_row_df.iloc[0].to_dict())
                labels.append(label)
            else:
                print(f"Cảnh báo: Không thể trích xuất đầy đủ features cho mẫu tại index {label_idx}. Bỏ qua mẫu này.")

    if not features_list:
        return pd.DataFrame(), pd.Series(dtype=str)

    # Đảm bảo DataFrame X có đúng thứ tự cột như đã định nghĩa
    X = pd.DataFrame(features_list, columns=FEATURE_COLUMNS)
    y = pd.Series(labels)
    return X, y

