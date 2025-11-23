import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import sys

# --- CẤU HÌNH TÊN FILE ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"

# --- KHỞI TẠO BIẾN TOÀN CỤC ---
df_movies = None
df_users = None
cosine_sim = None
LOGGED_IN_USER = None
MIN_POPULARITY_THRESHOLD = 0.5  # Ngưỡng độ phổ biến tối thiểu cho đề xuất


# ==============================================================================
# I. PHẦN TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================

def load_and_preprocess_data():
    """Tải và tiền xử lý dữ liệu cho cả hai hệ thống đề xuất."""
    global df_movies, df_users, cosine_sim
    try:
        # Tải dữ liệu phim
        df_movies = pd.read_csv(MOVIE_DATA_FILE).fillna("")
        df_movies.columns = [col.strip() for col in df_movies.columns]  # Làm sạch tên cột

        # Tải dữ liệu người dùng
        df_users = pd.read_csv(USER_DATA_FILE).fillna("")
        df_users.columns = [col.strip() for col in df_users.columns]

        # 1. Tiền xử lý cho Content-Based (TF-IDF/Cosine Sim)
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"] + " " +
                df_movies["Diễn viên chính"] + " " +
                df_movies["Thể loại phim"]
        )
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["combined_features"])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Chuẩn hóa Độ phổ biến (để dùng cho hệ thống TF-IDF)
        scaler = MinMaxScaler()
        df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])

        # 2. Tiền xử lý cho User-Based (Genre Matching)
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)

        return True

    except FileNotFoundError as e:
        print(f"\nLỖI FATAL: Không tìm thấy file dữ liệu: {e.filename}")
        print("Vui lòng đảm bảo các file CSV nằm cùng thư mục.")
        return False
    except KeyError as e:
        print(f"\nLỖI CỘT DỮ LIỆU: Thiếu cột {e}. Kiểm tra lại tên cột trong file CSV.")
        return False
    except Exception as e:
        print(f"\nLỖI KHÔNG XÁC ĐỊNH trong quá trình tải/tiền xử lý dữ liệu: {e}")
        return False


# ==============================================================================
# II. CHỨC NĂNG HỆ THỐNG GỢI Ý (CONTENT-BASED & USER-BASED)
# ==============================================================================

# --- A. Chức năng User-Based (Từ GoiYTuNguoiDungCu.py) ---
def parse_genres(genre_string):
    """Chuyển chuỗi thể loại thành tập hợp genres."""
    if not isinstance(genre_string, str) or not genre_string:
        return set()
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)


def get_recommendations(username, df_users, df_movies, num_recommendations=7):
    """
    Đề xuất phim dựa trên 5 phim người dùng xem gần nhất và sở thích thể loại.
    """
    user_row = df_users[df_users['Tên người dùng'] == username]

    # Lấy danh sách 5 phim đã xem gần nhất (Xử lý chuỗi list an toàn)
    try:
        watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
        watched_list = ast.literal_eval(watched_movies_str)
    except (ValueError, SyntaxError, IndexError):
        watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
        watched_list = [m.strip() for m in watched_movies_str.split(',') if m.strip()]

    favorite_movie = user_row['Phim yêu thích nhất'].iloc[0]
    watched_and_favorite = set(watched_list + [favorite_movie])

    # Xây dựng Hồ sơ Thể loại
    watched_genres = df_movies[df_movies['Tên phim'].isin(watched_list)]
    user_genres = set()
    for genres in watched_genres['parsed_genres']:
        user_genres.update(genres)

    if not user_genres:
        return pd.DataFrame()

    # Tính điểm đề xuất cho các phim CHƯA XEM
    candidate_movies = df_movies[~df_movies['Tên phim'].isin(watched_and_favorite)].copy()

    def calculate_score(candidate_genres):
        return len(candidate_genres.intersection(user_genres))

    candidate_movies['Similarity_Score'] = candidate_movies['parsed_genres'].apply(calculate_score)

    # Sắp xếp và Đề xuất
    recommended_df = candidate_movies.sort_values(
        by=['Similarity_Score', 'Độ phổ biến'],
        ascending=[False, False]
    )

    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'Similarity_Score']].head(num_recommendations)


# --- B. Chức năng Content-Based (Từ VeBD1.py) ---

def get_movie_index(movie_name):
    """Tìm chỉ mục của phim trong DataFrame."""
    try:
        idx = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()].index[0]
        return idx
    except IndexError:
        return -1


def recommend_movies_smart(movie_name, weight_sim=0.7, weight_pop=0.3):
    """
    Đề xuất phim dựa trên sự kết hợp giữa độ giống (sim) và độ phổ biến (pop).
    """
    idx = get_movie_index(movie_name)
    if idx == -1:
        print(f"Lỗi: Không tìm thấy phim '{movie_name}' trong dữ liệu.")
        return pd.DataFrame()

    # Tính toán điểm kết hợp
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'similarity'])

    # Kết hợp điểm similarity với độ phổ biến đã chuẩn hóa
    df_result = pd.merge(df_movies, sim_scores_df, left_index=True, right_on='index')

    # Tính điểm tổng hợp (Weighted Score)
    df_result['weighted_score'] = (
            weight_sim * df_result['similarity'] +
            weight_pop * df_result['popularity_norm']
    )

    # Loại bỏ phim đang xét
    df_result = df_result.drop(df_result[df_result['Tên phim'] == movie_name].index)

    # Sắp xếp theo điểm tổng hợp
    df_result = df_result.sort_values(by='weighted_score', ascending=False)

    return df_result[['Tên phim', 'weighted_score', 'similarity', 'Độ phổ biến']].head(10)


# ==============================================================================
# III. CHỨC NĂNG VẼ BIỂU ĐỒ & MENU
# ==============================================================================

def plot_genre_popularity(movie_name, top_movies, is_user_based=False):
    """Vẽ biểu đồ so sánh độ phổ biến của các thể loại và hiển thị.
    Sử dụng số thứ tự cho trục hoành (X-axis).
    """

    # 1. Lấy dữ liệu phim gốc (chỉ cần nếu là Content-Based) hoặc phim đã xem gần nhất (cho User-Based)
    genres_data = {}

    if is_user_based:
        global LOGGED_IN_USER
        user_row = df_users[df_users['Tên người dùng'] == LOGGED_IN_USER]
        watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
        watched_list = ast.literal_eval(watched_movies_str)

        # Thêm dữ liệu từ 5 phim đã xem gần nhất
        watched_df = df_movies[df_movies['Tên phim'].isin(watched_list)]
        for index, row in watched_df.iterrows():
            genres = row['Thể loại phim'].split(',')
            pop = row['Độ phổ biến']
            genres_data[f"{row['Tên phim']} (Đã xem)"] = {'genres': genres, 'pop': pop}

        title = f"Độ Phổ Biến của Các Thể Loại Phim Đề Xuất (Hồ sơ {LOGGED_IN_USER})"

    else:
        # Cho Content-Based
        movie_row = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()]
        if movie_row.empty:
            print(f"Lỗi: Không tìm thấy phim '{movie_name}' để vẽ biểu đồ.")
            return

        base_genres = movie_row['Thể loại phim'].iloc[0].split(',')

        # Thêm dữ liệu từ phim gốc
        genres_data[movie_name] = {'genres': base_genres, 'pop': movie_row['Độ phổ biến'].iloc[0]}
        title = f"Độ Phổ Biến của Các Thể Loại Phim Liên Quan đến '{movie_name}'"

    # 2. Lấy thể loại và độ phổ biến của các phim được đề xuất
    for index, row in top_movies.iterrows():
        genres = row['Thể loại phim'].split(',')
        pop = row['Độ phổ biến']
        genres_data[row['Tên phim']] = {'genres': genres, 'pop': pop}

    # 3. Tạo DataFrame cho biểu đồ
    plot_data = []
    for title_name, data in genres_data.items():
        for genre in data['genres']:
            plot_data.append({
                'Phim': title_name,
                'Thể loại': genre.strip(),
                'Độ phổ biến': data['pop']
            })

    df_plot = pd.DataFrame(plot_data)

    # 4. Vẽ biểu đồ
    plt.figure(figsize=(14, 7))  # Tăng kích thước biểu đồ

    # Lọc chỉ lấy các thể loại chính
    top_genres = df_plot['Thể loại'].value_counts().nlargest(7).index.tolist()
    df_plot_filtered = df_plot[df_plot['Thể loại'].isin(top_genres)]

    # Sắp xếp thể loại theo độ phổ biến trung bình để đảm bảo thứ tự
    genre_avg_pop = df_plot_filtered.groupby('Thể loại')['Độ phổ biến'].mean().sort_values(
        ascending=False).index.tolist()

    # Tạo mapping từ chỉ số sang tên thể loại để chú giải
    genre_map = {i + 1: genre for i, genre in enumerate(genre_avg_pop)}

    # Tạo màu sắc cho từng phim
    unique_movies = df_plot_filtered['Phim'].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_movies))
    movie_color_map = {movie: colors(i) for i, movie in enumerate(unique_movies)}

    # Vẽ Bar cho từng phim
    bar_width = 0.8 / len(unique_movies)

    # Chú thích giải thích số trên trục X
    legend_text = "\n\nCHÚ THÍCH TRỤC X:\n" + "\n".join([f"{idx}: {genre}" for idx, genre in genre_map.items()])

    # Dùng chỉ số số học làm vị trí X
    x_pos = np.arange(len(genre_avg_pop))  # Vị trí chính giữa cho từng nhóm thể loại

    # Dùng numpy để tính toán vị trí cho các thanh bar trong cùng một nhóm
    for i, genre in enumerate(genre_avg_pop):
        genre_data = df_plot_filtered[df_plot_filtered['Thể loại'] == genre].sort_values(by='Độ phổ biến',
                                                                                         ascending=False)

        num_movies_in_genre = len(genre_data)

        # Tính toán vị trí offset cho từng thanh bar trong nhóm
        offsets = np.linspace(-bar_width * (num_movies_in_genre / 2), bar_width * (num_movies_in_genre / 2),
                              num_movies_in_genre, endpoint=False) + bar_width / 2

        for j, (idx, row) in enumerate(genre_data.iterrows()):
            plt.bar(x_pos[i] + offsets[j], row['Độ phổ biến'],
                    width=bar_width,
                    color=movie_color_map[row['Phim']],
                    alpha=0.8)

    # Cài đặt nhãn trục X chỉ là số
    plt.xticks(x_pos, [str(idx) for idx in genre_map.keys()], fontsize=12)

    plt.xlabel("Thể loại (Tham chiếu số ở dưới)")
    plt.ylabel("Độ Phổ Biến (Popularity Score)")

    # Thêm chú giải văn bản cho trục X bên dưới biểu đồ
    plt.figtext(0.5, -0.05, legend_text, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

    # Tạo custom legend (chú giải) theo Tên Phim
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=movie_color_map[movie]) for movie in unique_movies]
    plt.legend(legend_handles, unique_movies, title="Phim", loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 0.85, 1])  # Điều chỉnh layout để chú giải không bị che và nhãn trục X hiển thị
    plt.grid(axis='y', linestyle='--')

    plt.show()

    print("✅ Biểu đồ so sánh thể loại đã được hiển thị trên màn hình.")


def display_main_menu():
    """Hiển thị menu chính."""
    global LOGGED_IN_USER
    print("\n" + "=" * 50)
    print(f"🎬 HỆ THỐNG ĐỀ XUẤT PHIM (Người dùng: {LOGGED_IN_USER})")
    print("=" * 50)
    print("1. Tìm kiếm và đề xuất phim theo TÊN (Cosine Sim + Pop)")
    # ĐÃ SỬA TỪ 9 THÀNH 2
    print("2. Đề xuất phim theo HỒ SƠ NGƯỜI DÙNG (5 phim coi gần nhất)")
    print("0. Đăng xuất / Thoát chương trình")
    print("-" * 50)


def user_login():
    """Xử lý đăng nhập/chọn người dùng trước menu."""
    global LOGGED_IN_USER
    while True:
        sample_users = df_users['Tên người dùng'].head(5).tolist()
        print("\n" + "=" * 50)
        print("🤝 CHỌN NGƯỜI DÙNG ĐĂNG NHẬP")
        print("=" * 50)
        print(f"* Thử nghiệm với các tên sau: {', '.join(sample_users)}, ...")

        username = input("▶️ Vui lòng nhập Tên người dùng cần đăng nhập: ").strip()

        if username.lower() == 'thoat':
            return False

        if username in df_users['Tên người dùng'].values:
            LOGGED_IN_USER = username
            print(f"\n✅ Chào mừng, {LOGGED_IN_USER}!")
            return True
        else:
            print(f"❌ Tên người dùng '{username}' không tồn tại. Vui lòng thử lại hoặc nhập 'thoat'.")


def main_app():
    """Chức năng chính của ứng dụng."""

    if not load_and_preprocess_data():
        return

    if not user_login():
        print("Đã thoát chương trình.")
        return

    while True:
        display_main_menu()
        choice = input("👉 Nhập lựa chọn của bạn: ").strip()

        if choice == "0":
            print(f"\nĐã đăng xuất khỏi {LOGGED_IN_USER}. Chương trình kết thúc.")
            break

        elif choice == "1":
            movie_name = input("🎥 Nhập tên phim bạn yêu thích: ").strip()
            if not movie_name:
                continue

            try:
                weight_sim = float(input("⚖️ Trọng số độ giống (0-1, mặc định 0.7): ") or 0.7)
                weight_pop = 1 - weight_sim
            except ValueError:
                weight_sim, weight_pop = 0.7, 0.3

            result = recommend_movies_smart(movie_name, weight_sim, weight_pop)

            if not result.empty:
                print(f"\n🎬 10 Đề xuất phim dựa trên '{movie_name}':")
                print(result.to_markdown(index=False))

                if input("\n📊 Bạn có muốn vẽ biểu đồ so sánh thể loại? (y/n): ").lower() == "y":
                    # Vẽ biểu đồ cho Content-Based
                    plot_genre_popularity(movie_name,
                                          df_movies[df_movies['Tên phim'].isin(result['Tên phim'].tolist())],
                                          is_user_based=False)
            else:
                print("⚠️ Không tìm thấy đề xuất hoặc phim gốc không tồn tại.")

            input("\nNhấn Enter để tiếp tục...")

        elif choice == "2":  # ĐÃ SỬA TỪ 9 THÀNH 2
            print(f"\n--- ĐANG ĐỀ XUẤT PHIM CHO {LOGGED_IN_USER} (Dựa trên 5 phim gần nhất) ---")

            # 1. Hiển thị 5 phim đã xem gần nhất
            recent_films = df_users[df_users['Tên người dùng'] == LOGGED_IN_USER]['5 phim coi gần nhất'].iloc[0]
            print(f"5 Phim đã xem gần nhất: {recent_films}")

            # 2. Chạy hàm đề xuất User-Based
            recommendations = get_recommendations(LOGGED_IN_USER, df_users, df_movies, num_recommendations=10)

            # 3. In kết quả
            if not recommendations.empty:
                print("\n✅ 10 Đề xuất Phim Dành Cho Bạn (Ưu tiên Thể loại & Độ phổ biến):")
                print(recommendations.to_markdown(index=False))

                if input("\n📊 Bạn có muốn vẽ biểu đồ so sánh thể loại? (y/n): ").lower() == "y":
                    # Vẽ biểu đồ cho User-Based
                    # Truyền kết quả đề xuất và đặt cờ is_user_based=True
                    plot_genre_popularity(None,  # Không cần movie_name khi là user-based
                                          df_movies[df_movies['Tên phim'].isin(recommendations['Tên phim'].tolist())],
                                          is_user_based=True)
            else:
                print("⚠️ Không có đề xuất nào được tạo. Kiểm tra dữ liệu thể loại phim đã xem.")

            input("\nNhấn Enter để tiếp tục...")

        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng nhập 1, 2, hoặc 0.")


if __name__ == "__main__":
    main_app()