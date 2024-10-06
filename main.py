import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, FalconForCausalLM, AutoTokenizer  # Sửa ở đây

# Hàm lấy loại mô hình mà người dùng muốn
def get_model_type():
    print("Các loại mô hình có sẵn:")
    print("1. Classification")
    print("2. Regression")
    print("3. NLP")
    print("4. LLM (Large Language Model)")
    choice = input("Nhập loại mô hình bạn muốn sử dụng (ví dụ: Classification, Regression, NLP, LLM): ").lower()
    return choice

# Tự động vẽ biểu đồ cho dataset
def visualize_data(df):
    print("Đang vẽ biểu đồ...")

    df.hist(bins=30, figsize=(10, 8))
    plt.tight_layout()
    plt.show()

    if len(df.columns) > 1:
        sns.pairplot(df)
        plt.show()

    if df.select_dtypes(include='number').shape[1] > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Heatmap of Correlation")
        plt.show()

# Kiểm tra và phân tích sơ bộ dữ liệu
def analyze_data(df):
    print("Phân tích sơ bộ dữ liệu:")
    print(df.describe())
    print("\nKiểm tra dữ liệu thiếu:")
    print(df.isnull().sum())

# Huấn luyện mô hình RNN
def train_rnn(X_train, y_train, input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128))
    model.add(SimpleRNN(64))
    model.add(Dense(1, activation='sigmoid'))  # Hoặc 'softmax' cho multi-class
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Thay đổi loss function nếu cần
    model.fit(X_train, y_train, epochs=10, batch_size=32)

# Chọn model phù hợp dựa trên loại do người dùng chọn
def suggest_and_train_model(df, model_type):
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_type == 'classification':
        print(f"Đề xuất các model Classification: RandomForest, SVM, LogisticRegression")
        model_choice = input("Chọn model bạn muốn (RandomForest, SVM, LogisticRegression): ").lower()
        if model_choice == 'randomforest':
            model = RandomForestClassifier()
        elif model_choice == 'svm':
            model = SVC()
        elif model_choice == 'logisticregression':
            model = LogisticRegression()
        else:
            print("Model không hợp lệ!")
            return
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Độ chính xác của model {model_choice}: {accuracy:.2f}")

    elif model_type == 'regression':
        print(f"Đề xuất các model Regression: LinearRegression, SVR, RandomForestRegressor")
        model_choice = input("Chọn model bạn muốn (LinearRegression, SVR, RandomForestRegressor): ").lower()
        if model_choice == 'linearregression':
            model = LinearRegression()
        elif model_choice == 'svr':
            model = SVR()
        elif model_choice == 'randomforestregressor':
            model = RandomForestRegressor()
        else:
            print("Model không hợp lệ!")
            return
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error của model {model_choice}: {mse:.2f}")

    elif model_type == 'nlp':
        print(f"Đề xuất model NLP: RNN")
        model_choice = input("Chọn model NLP bạn muốn (RNN): ").lower()
        if model_choice == 'rnn':
            # Chuyển đổi dữ liệu cho RNN
            X_train_rnn = X_train.values.reshape(-1, X_train.shape[1], 1)  # Reshape cho RNN
            train_rnn(X_train_rnn, y_train, input_dim=X.shape[1])
        else:
            print("Model không hợp lệ!")

    elif model_type == 'llm':
        print(f"Đề xuất model LLM: LLaMA, Falcon")
        model_choice = input("Chọn model LLM bạn muốn (LLaMA, Falcon): ").lower()
        if model_choice == 'llama':
            # Khởi tạo model LLaMA
            tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
            model = LlamaForCausalLM.from_pretrained('huggyllama/llama-7b')
        elif model_choice == 'falcon':
            # Khởi tạo model Falcon
            tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b')  # Sửa ở đây
            model = FalconForCausalLM.from_pretrained('tiiuae/falcon-7b')
        else:
            print("Model không hợp lệ!")
            return

        print(f"Model LLM được chọn: {model_choice}. Tạm thời không triển khai model này trong ví dụ này.")

    else:
        print("Loại model không hợp lệ!")

# Hàm chính để chạy tool
def run_tool():
    Tk().withdraw()
    file_path = askopenfilename(title="Chọn file dataset", filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])

    if not file_path:
        print("Bạn chưa chọn file!")
        return

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            print("Chỉ hỗ trợ file CSV và Excel!")
            return
    except Exception as e:
        print(f"Lỗi khi đọc dataset: {e}")
        return

    analyze_data(df)
    visualize_data(df)

    model_type = get_model_type()
    suggest_and_train_model(df, model_type)

# Chạy tool
if __name__ == "__main__":
    run_tool()
